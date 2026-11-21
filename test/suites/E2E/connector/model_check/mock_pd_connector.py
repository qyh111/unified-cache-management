# SPDX-License-Identifier: MIT
"""Lightweight P/D sibling connector for MultiConnector handshake testing.

Mimics a real Ascend P/D transfer connector (such as MooncakeConnector) placed
ahead of UCMConnector inside vLLM's MultiConnector / AscendMultiConnector:
- Advertises non-None KVConnectorHandshakeMetadata on every worker rank.
- Validates that every entry delivered to set_xfer_handshake_metadata_pp_aware()
  belongs to this connector (catching cross-child metadata mixups).
- Performs no local KV hits or writes so MultiConnector delegates external
  prefix hits and dumps to the sibling UCMConnector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class MockPDHandshakeMetadata(KVConnectorHandshakeMetadata):
    """Non-dict handshake payload matching real MooncakeHandshakeMetadata shape."""

    engine_id: str
    world_rank: int
    endpoint: str


class MockPDConnectorMetadata(KVConnectorMetadata):
    """Empty per-step metadata for the no-op P/D sibling connector."""


class MockPDHandshakeConnector(KVConnectorBase_V1, SupportsHMA):
    """Minimal P/D transfer connector with worker->scheduler handshake metadata."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._role = role
        self._engine_id = str(
            getattr(vllm_config.kv_transfer_config, "engine_id", "mock-pd")
        )
        self._world_rank = int(getattr(vllm_config.parallel_config, "rank", 0))
        self.received_handshake_metadata: dict[Any, Any] | None = None

    # ---- Worker-side SPI ---------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        return None

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        if self._role != KVConnectorRole.WORKER:
            return None
        return MockPDHandshakeMetadata(
            engine_id=self._engine_id,
            world_rank=self._world_rank,
            endpoint=f"mock-pd://rank-{self._world_rank}",
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    # ---- Scheduler-side SPI ------------------------------------------------

    def _validate_handshake_payload(self, metadata: dict[Any, Any]) -> None:
        if not metadata:
            raise ValueError("MockPDHandshakeConnector received empty handshake metadata")
        for key, value in metadata.items():
            if not isinstance(value, MockPDHandshakeMetadata):
                raise TypeError(
                    "MockPDHandshakeConnector expected MockPDHandshakeMetadata at "
                    f"{key!r}, got {type(value).__name__}"
                )
        self.received_handshake_metadata = dict(metadata)

    def set_xfer_handshake_metadata(self, metadata: dict[int, Any]) -> None:
        self._validate_handshake_payload(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, ...], Any]
    ) -> None:
        self._validate_handshake_payload(metadata)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        return None

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return MockPDConnectorMetadata()

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None
