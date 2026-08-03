"""No-I/O vLLM connector for forward and per-attention timing."""

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
except ImportError:
    KVConnectorWorkerMetadata = object

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from ucm.integration.vllm.device import Device, create_device
from ucm.logger import init_logger
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def inference_duration_monitor_enabled(extra_config: dict[str, Any]) -> bool:
    """Resolve the monitor switch from inline or YAML-backed UCM config."""
    kv_transfer_config = SimpleNamespace(kv_connector_extra_config=extra_config)
    launch_config = Config(kv_transfer_config).get_config() or {}
    return bool(launch_config.get("use_inference_duration_monitor", False))


@dataclass
class InferenceDurationMonitorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for a no-I/O monitor step."""

    preempted_req_ids: set[str] = field(default_factory=set)
    scheduled_reqs: int = 0
    new_reqs: int = 0
    new_reqs_with_computed_tokens: int = 0
    scheduled_tokens: int = 0
    total_num_computed_tokens: int = 0

    @property
    def should_collect_duration(self) -> bool:
        return self.new_reqs_with_computed_tokens > 0


@dataclass
class DurationStats:
    """Mergeable duration summary in milliseconds."""

    count: int = 0
    sum_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def observe(self, value_ms: float) -> None:
        value_ms = float(value_ms)
        self.count += 1
        self.sum_ms += value_ms
        self.min_ms = min(self.min_ms, value_ms)
        self.max_ms = max(self.max_ms, value_ms)

    def aggregate(self, other: "DurationStats") -> None:
        if other.count == 0:
            return
        self.count += other.count
        self.sum_ms += other.sum_ms
        self.min_ms = min(self.min_ms, other.min_ms)
        self.max_ms = max(self.max_ms, other.max_ms)

    @property
    def avg_ms(self) -> float:
        return self.sum_ms / self.count if self.count else 0.0


@dataclass
class InferenceDurationMonitorWorkerMetadata(KVConnectorWorkerMetadata):
    """Per-forward timing data aggregated across one DP engine's workers."""

    duration_stats: dict[str, DurationStats] = field(default_factory=dict)
    worker_ranks: set[int] = field(default_factory=set)

    def aggregate(self, other: Any) -> Any:
        assert isinstance(other, InferenceDurationMonitorWorkerMetadata)
        for name, other_stats in other.duration_stats.items():
            self.duration_stats.setdefault(name, DurationStats()).aggregate(other_stats)
        self.worker_ranks.update(other.worker_ranks)
        return self


class UCMInferenceDurationMonitorConnector(KVConnectorBase_V1, SupportsHMA):
    """Measure forward and attention duration without performing KV I/O.

    The layer hooks surround vLLM's attention invocation, so per-layer values
    describe attention device time rather than the complete transformer block.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        parallel_config = vllm_config.parallel_config
        self._dp_rank = int(getattr(parallel_config, "data_parallel_rank", 0))
        self._model_rank = int(getattr(parallel_config, "rank", 0))
        self._hbm_hit_tokens_by_request: dict[str, int] = {}
        self._device: Optional[Device] = None
        self._collect_current_forward = False
        self._inference_start_time: Optional[float] = None
        self._active_attention_events: dict[str, Any] = {}
        self._pending_attention_events: list[tuple[str, Any, Any]] = []
        self._duration_stats: dict[str, DurationStats] = {}
        self._pending_worker_metadata: Optional[
            InferenceDurationMonitorWorkerMetadata
        ] = None
        logger.info("Init UCMInferenceDurationMonitorConnector (no KV I/O).")

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        del extra_config
        return True

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        del kv_caches
        self._device = self._create_device()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        request_tokens = getattr(request, "num_tokens", None)
        if request_tokens is None:
            request_tokens = len(getattr(request, "all_token_ids", ()))
        self._hbm_hit_tokens_by_request[request.request_id] = min(
            max(int(num_computed_tokens), 0), max(int(request_tokens), 0)
        )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        del request, blocks, num_external_tokens

    @staticmethod
    def _scheduled_request_ids(scheduler_output: SchedulerOutput) -> list[str]:
        request_ids = [
            request.req_id for request in scheduler_output.scheduled_new_reqs
        ]
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if isinstance(cached_reqs, list):
            request_ids.extend(request.req_id for request in cached_reqs)
        else:
            request_ids.extend(getattr(cached_reqs, "req_ids", ()))
        return request_ids

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        scheduled_request_ids = list(
            getattr(scheduler_output, "num_scheduled_tokens", {}).keys()
        )
        if not scheduled_request_ids:
            scheduled_request_ids = self._scheduled_request_ids(scheduler_output)
        new_requests = scheduler_output.scheduled_new_reqs
        new_request_num_computed_tokens = [
            max(
                int(
                    getattr(
                        request,
                        "num_computed_tokens",
                        self._hbm_hit_tokens_by_request.get(request.req_id, 0),
                    )
                ),
                0,
            )
            for request in new_requests
        ]
        new_reqs_with_computed_tokens = sum(
            num_computed_tokens > 0
            for num_computed_tokens in new_request_num_computed_tokens
        )
        total_num_computed_tokens = sum(new_request_num_computed_tokens)
        scheduled_tokens = sum(
            int(tokens)
            for tokens in getattr(scheduler_output, "num_scheduled_tokens", {}).values()
        )
        logger.info(
            "Inference duration scheduler stats: rank=%d, "
            "scheduled_reqs=%d, new_reqs=%d, "
            "new_reqs_with_computed_tokens=%d, scheduled_tokens=%d, "
            "total_num_computed_tokens=%d",
            self._model_rank,
            len(scheduled_request_ids),
            len(new_requests),
            new_reqs_with_computed_tokens,
            scheduled_tokens,
            total_num_computed_tokens,
        )
        for request_id in getattr(scheduler_output, "finished_req_ids", ()):
            self._hbm_hit_tokens_by_request.pop(request_id, None)
        return InferenceDurationMonitorMetadata(
            preempted_req_ids=scheduler_output.preempted_req_ids or set(),
            scheduled_reqs=len(scheduled_request_ids),
            new_reqs=len(new_requests),
            new_reqs_with_computed_tokens=new_reqs_with_computed_tokens,
            scheduled_tokens=scheduled_tokens,
            total_num_computed_tokens=total_num_computed_tokens,
        )

    @staticmethod
    def _create_device() -> Device:
        device = create_device()
        if device is None:
            raise RuntimeError(
                "Unsupported device platform for inference duration monitoring."
            )
        return device

    def _get_device(self) -> Device:
        if self._device is None:
            self._device = self._create_device()
        return self._device

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, InferenceDurationMonitorMetadata)
        self._collect_current_forward = metadata.should_collect_duration
        self._inference_start_time = None
        self._active_attention_events.clear()
        self._pending_attention_events.clear()
        self._duration_stats.clear()
        if not self._collect_current_forward:
            return
        self._get_device().synchronize()
        self._inference_start_time = time.perf_counter()

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._collect_current_forward:
            return
        try:
            self._active_attention_events[layer_name] = (
                self._get_device().record_timing_event()
            )
        except Exception as error:
            logger.warning(
                "Failed to record attention start event for %s: %s",
                layer_name,
                error,
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        del kv_layer, attn_metadata, kwargs
        if not self._collect_current_forward:
            return
        start_event = self._active_attention_events.pop(layer_name, None)
        if start_event is None:
            return
        try:
            end_event = self._get_device().record_timing_event()
        except Exception as error:
            logger.warning(
                "Failed to record attention end event for %s: %s",
                layer_name,
                error,
            )
            return
        self._pending_attention_events.append((layer_name, start_event, end_event))

    def _observe_duration(self, name: str, value_ms: float) -> None:
        self._duration_stats.setdefault(name, DurationStats()).observe(value_ms)

    def wait_for_save(self) -> None:
        if self._inference_start_time is None:
            self._collect_current_forward = False
            return
        device = self._get_device()
        device.synchronize()
        elapsed_ms = (time.perf_counter() - self._inference_start_time) * 1000
        self._inference_start_time = None
        self._collect_current_forward = False
        self._observe_duration("forward", elapsed_ms)

        if not self._pending_attention_events:
            logger.warning_once(
                "Inference duration monitor observed no attention hooks. "
                "Per-attention timing is unavailable when the active model "
                "execution path bypasses KV connector layer hooks."
            )
        for layer_name, start_event, end_event in self._pending_attention_events:
            try:
                layer_elapsed_ms = device.elapsed_time_ms(start_event, end_event)
            except Exception as error:
                logger.warning(
                    "Failed to read attention duration for %s: %s",
                    layer_name,
                    error,
                )
                continue
            self._observe_duration(f"attention_layer:{layer_name}", layer_elapsed_ms)

        self._active_attention_events.clear()
        self._pending_attention_events.clear()
        self._pending_worker_metadata = InferenceDurationMonitorWorkerMetadata(
            duration_stats=self._duration_stats,
            worker_ranks={self._model_rank},
        )
        self._duration_stats = {}
        logger.info(
            "Inference duration monitor: dp_rank=%d, model_rank=%d, "
            "start_load_kv_to_wait_for_save_ms=%.3f",
            self._dp_rank,
            self._model_rank,
            elapsed_ms,
        )

    def build_connector_worker_meta(
        self,
    ) -> Optional[InferenceDurationMonitorWorkerMetadata]:
        metadata = self._pending_worker_metadata
        self._pending_worker_metadata = None
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        metadata = getattr(connector_output, "kv_connector_worker_meta", None)
        if not isinstance(metadata, InferenceDurationMonitorWorkerMetadata):
            return
        for name in sorted(metadata.duration_stats):
            stats = metadata.duration_stats[name]
            if stats.count == 0:
                continue
            logger.info(
                "Inference duration aggregate: dp_rank=%d, workers=%d, "
                "scope=%s, count=%d, avg_ms=%.3f, min_ms=%.3f, max_ms=%.3f",
                self._dp_rank,
                len(metadata.worker_ranks),
                name,
                stats.count,
                stats.avg_ms,
                stats.min_ms,
                stats.max_ms,
            )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del request, block_ids
        return False, None
