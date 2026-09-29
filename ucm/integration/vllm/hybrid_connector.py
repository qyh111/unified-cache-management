"""Single-store FA / Mamba connector with v2 physical layout construction.

UCMDirectConnector remains the owner of v1 event, wait, failure-reporting and
preemption helpers. No new store method or publication protocol is introduced.
"""

import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
)
from vllm.platforms import current_platform

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.hybrid.scheduler import UCMDispatcher
from ucm.integration.vllm.hybrid.spec import parse_kv_cache_config
from ucm.integration.vllm.hybrid.store_layout import HybridStoreLayout, validate_spec
from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.integration.vllm.ucm_connector import (
    PendingDumpTask,
    UCMDirectConnector,
    _get_store_io_sizes,
    _scheduler_read_block_size,
    _use_ucm_connector_cpu_affinity,
)
from ucm.logger import init_logger
from ucm.store.factory_v1 import UcmConnectorFactoryV1

logger = init_logger(__name__)


class _StoreLookup:
    def __init__(self, connector):
        self.connector = connector

    def lookup_on_prefix(self, keys):
        c = self.connector
        return c._rank_consistency.lookup_on_prefix(c.store, list(keys)) if keys else -1

    def lookup_on_reverse(self, keys):
        c = self.connector
        return (
            c._rank_consistency.lookup_on_reverse(c.store, list(keys)) if keys else -1
        )


class UCMHybridConnector(UCMDirectConnector, SupportsHMA):
    _defer_scheduler_store = True

    @staticmethod
    def _consistency_manager_enabled(launch_config, is_mla):
        # Hybrid scopes all TP ranks independently, including state with MLA.
        return launch_config.get("use_consistency_manager", True)

    def __init__(self, vllm_config, role, kv_cache_config=None):
        parallel = vllm_config.parallel_config
        for name in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if int(getattr(parallel, name, 1)) != 1:
                raise ValueError(
                    f"Hybrid v1-store bring-up requires {name}=1; keep the existing connector for this topology"
                )
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            # All TP ranks dump here, even for MLA. Reuse the existing
            # intersection aggregation rather than the MLA rank-0-only union.
            self._connector_worker_meta.is_mla = False
        self.use_layerwise = bool(self.launch_config.get("use_layerwise", True))
        if self.launch_config.get("use_request_async_load", False):
            raise ValueError("Hybrid does not yet support request-async loading")
        if (
            len(self.connector_configs) != 1
            or self.connector_configs[0]["ucm_connector_name"] != "UcmPipelineStore"
        ):
            raise ValueError("Hybrid requires exactly one UcmPipelineStore")
        config = self.connector_configs[0]["ucm_connector_config"]
        if config.get("store_pipeline") not in ("Cache|Posix", "Cache|Empty"):
            raise ValueError(
                "Hybrid padding is currently checked for Cache|Posix and Cache|Empty only"
            )
        if config.get("tensor_size", 0):
            raise ValueError(
                "Hybrid supplies tensor_size_list; remove the scalar tensor_size override"
            )
        text_config = vllm_config.model_config.hf_text_config
        # Runtime groups can be enlarged by the engine (e.g. Mamba align),
        # independently of the launch CacheConfig block_size.
        scheduler_block = math.lcm(
            *(
                int(group.kv_cache_spec.block_size)
                for group in kv_cache_config.kv_cache_groups
            )
        )
        self.spec = parse_kv_cache_config(
            kv_cache_config,
            scheduler_block_size=scheduler_block,
            device_type=current_platform.device_type,
            num_hidden_layers=int(text_config.num_hidden_layers),
        )
        validate_spec(self.spec)
        configured_block = self.launch_config.get(
            "ucm_cache_block_size", self.spec.ucm_cache_block_size
        )
        if int(configured_block) != self.spec.ucm_cache_block_size:
            raise ValueError(
                "Hybrid requires UCM block size equal to group token_block_size"
            )
        self.block_size = self.hash_block_size = self.spec.ucm_cache_block_size
        self.blocks_per_chunk = 1
        # A separate, versioned namespace is necessary: padded Hybrid records
        # are not byte-compatible with Direct/HLA or the v2 reference Proxy.
        identity = {
            "device": current_platform.device_type,
            "cache_dtype": str(vllm_config.cache_config.cache_dtype),
            "model": (
                text_config.to_dict()
                if hasattr(text_config, "to_dict")
                else vars(text_config)
            ),
            "tp": self.tp_size,
        }
        for package in ("vllm", "vllm-ascend"):
            try:
                identity[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                identity[package] = "unknown"
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        self._namespace = (
            f"hybrid-v1-r1-{digest}-b{self.block_size}-lw{int(self.use_layerwise)}"
        )
        self.kv_cache_layout = None
        self._load_tasks = {}
        self._seen_names = set()
        self._saved_rows = set()
        self._failed_load_reqs = set()
        self._save_complete = False
        self._dump_request_ids = set()
        self.dispatcher = None
        if role == KVConnectorRole.SCHEDULER:
            self.store = self._create_store(None)
            hasher = RequestHasher(vllm_config, 0)
            self.dispatcher = UCMDispatcher(
                self.spec,
                _StoreLookup(self),
                hasher,
                hasher((self._namespace, str(vllm_config.cache_config.cache_dtype))),
                load_threshold_tokens=self.launch_config.get(
                    "load_tokens_threshold", 0
                ),
                recompute_tokens=self._get_full_hit_recompute_tokens(),
            )

    def _create_store(self, kv_cache_layout, cpu_affinity_cores=None):
        config = copy.deepcopy(self.connector_configs[0]["ucm_connector_config"])
        paths = config.get("storage_backends", [])
        if isinstance(paths, str):
            paths = paths.split(os.pathsep)
        config["storage_backends"] = [
            os.path.join(path, self._namespace) for path in paths
        ]
        for path in config["storage_backends"]:
            os.makedirs(path, exist_ok=True)
        # Different TP ranks have different keys. Keep their in-memory buffers
        # separate too; no assumption that MLA implies identical Mamba states.
        config.setdefault("share_buffer_enable", False)
        self._set_default_shm_buffer_capacity(config)
        config["unique_id"] = f"{self.unique_id}_{self._namespace}_rank{self.tp_rank}"
        config["local_rank_size"] = 1
        config["posix_gc_enable"] = self._gc_owner
        if kv_cache_layout is not None:
            shard_size, block_size = _get_store_io_sizes(
                kv_cache_layout.shard_size, kv_cache_layout.block_size
            )
            config.update(
                device_id=self.device_id,
                tensor_size_list=kv_cache_layout.tensor_size_list,
                shard_size=shard_size,
                block_size=block_size,
                gpu_kv_buffer_addrs=kv_cache_layout.base_ptrs.tolist(),
                gpu_kv_buffer_sizes=kv_cache_layout.buffer_sizes.tolist(),
            )
            self._publish_block_size(block_size)
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
            logger.info(
                "Hybrid store schema: rows=%s tensor_size_list=%s shard_size=%s block_size=%s",
                kv_cache_layout.row_count,
                config["tensor_size_list"],
                shard_size,
                block_size,
            )
        elif self._gc_owner:
            size = _scheduler_read_block_size()
            if size is None:
                # An estimate based on head dimensions cannot describe padded
                # Mamba/indexer records. Let the user specify it for GC only.
                size = config.get("block_size")
            if size is None:
                raise ValueError(
                    "Hybrid GC requires worker-published or configured block_size"
                )
            config["block_size"] = size
        return UcmConnectorFactoryV1.create_connector("UcmPipelineStore", config)

    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches
        self.kv_cache_layout = HybridStoreLayout(
            self.spec, kv_caches, layerwise=self.use_layerwise
        )
        self.layer_name_to_id = self.kv_cache_layout.layer_name_to_id
        self.block_data_size = self.kv_cache_layout.block_size
        self.device = create_device()
        if self.device is None:
            raise RuntimeError("Hybrid requires CUDA, Ascend, or explicit CPU Model-check simulation")
        worker_cores, store_cores = (
            self.device.split_cores(self.device_id)
            if _use_ucm_connector_cpu_affinity()
            else (None, None)
        )
        self.store = self._create_store(self.kv_cache_layout, store_cores)
        diagnostic_dir = os.environ.get("UCM_HYBRID_DUMP_LAYOUT")
        if diagnostic_dir:
            root = Path(diagnostic_dir)
            root.mkdir(parents=True, exist_ok=True)
            schema = {
                "namespace": self._namespace,
                "rank": self.tp_rank,
                "tensor_size_list": self.kv_cache_layout.tensor_size_list,
                "ucm_block_offsets": self.kv_cache_layout.ucm_block_offsets.tolist(),
                "groups": [
                    {
                        "group_id": group.group_id,
                        "token_block_size": group.token_block_size,
                        "layers": [layer.layer_name for layer in group.layers],
                    }
                    for group in self.spec.groups
                ],
                "rows": {
                    kind: [
                        {
                            "layer_id": row.layer_id,
                            "names": sorted(row.names),
                            "columns": row.columns.tolist(),
                            "block_strides": row.strides.tolist(),
                        }
                        for row in rows
                    ]
                    for kind, rows in self.kv_cache_layout.rows.items()
                },
            }
            (root / f"hybrid-layout-rank{self.tp_rank}.json").write_text(
                json.dumps(schema, indent=2), encoding="utf-8"
            )
        if worker_cores:
            os.sched_setaffinity(0, worker_cores)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if self.persist_token_threshold > request.num_tokens:
            self.dispatcher.requests.pop(str(request.request_id), None)
            return 0, False
        result = self.dispatcher.lookup(request, num_computed_tokens)
        return result.external_hit_tokens, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        # SchedulerOutput carries complete tables and allocation deltas.
        return None

    def build_connector_meta(self, scheduler_output):
        return self.dispatcher.build_from_scheduler_output(scheduler_output)

    def _store_keys(self, keys):
        return (
            list(keys)
            if self.tp_rank == 0
            else [self.request_hasher(key) for key in keys]
        )

    def _mark_load_failed(self, request_id, request):
        self._failed_load_reqs.add(request_id)
        self._connector_worker_meta.mark_failed(request_id)
        self._invalid_block_ids.update(
            int(block)
            for plan in request.load_plans
            for window in plan.windows
            for block in window
            if block
        )

    def _wait_layer(self, layer_id):
        metadata = self._get_connector_metadata()
        for request_id, task in self._load_tasks.pop(layer_id, []):
            try:
                self._rank_consistency.wait_load(task)
            except Exception:
                logger.exception("Hybrid load wait failed for %s", request_id)
                self._mark_load_failed(request_id, metadata.requests[request_id])

    def start_load_kv(self, forward_context, **kwargs):
        if self._load_tasks or self._pending_dump_tasks:
            raise RuntimeError("Previous Hybrid transfers have not been drained")
        self._seen_names.clear()
        self._saved_rows.clear()
        self._failed_load_reqs.clear()
        self._dump_request_ids.clear()
        self._save_complete = False
        if not self.has_connector_metadata():
            return
        # Mamba preprocess may have zeroed/reused the destination on compute.
        self.device.synchronize()
        for request_id, request in self._get_connector_metadata().requests.items():
            for plan in request.load_plans:
                for index, row in enumerate(self.kv_cache_layout.rows[plan.hash_group]):
                    try:
                        transfer = self.kv_cache_layout.resolve(plan, index)
                        if not transfer.keys:
                            raise ValueError(
                                "Load plan refers to an absent state snapshot"
                            )
                        task = self._rank_consistency.submit_load(
                            self.store,
                            {request_id: list(transfer.keys)},
                            self._store_keys(transfer.keys),
                            [index] * len(transfer.keys),
                            transfer.ptrs,
                        )
                        self._load_tasks.setdefault(row.layer_id, []).append(
                            (request_id, task)
                        )
                    except Exception:
                        logger.exception("Hybrid load submit failed for %s", request_id)
                        self._mark_load_failed(request_id, request)
            # State layers do not necessarily have attention hooks.
        state_layers = {
            layer.layer_index
            for group in self.spec.state_groups
            for layer in group.layers
        }
        for layer_id in tuple(self._load_tasks):
            if not self.use_layerwise or layer_id is None or layer_id in state_layers:
                self._wait_layer(layer_id)

    def wait_for_layer_load(self, layer_name):
        if self.has_connector_metadata():
            self._wait_layer(self.layer_name_to_id.get(layer_name))

    def _save_row(self, kind, row_index):
        identity = (kind, row_index)
        if identity in self._saved_rows:
            return
        self._saved_rows.add(identity)
        for request_id, request in self._get_connector_metadata().requests.items():
            if request_id in self._failed_load_reqs:
                continue
            for plan in request.dump_plans:
                if plan.hash_group != kind:
                    continue
                transfer = self.kv_cache_layout.resolve(plan, row_index)
                if not transfer.keys:
                    continue
                event = 0
                try:
                    event = self._get_dump_event_handle()
                    task = self._rank_consistency.submit_dump(
                        self.store,
                        {request_id: set(transfer.keys)},
                        self._store_keys(transfer.keys),
                        [row_index] * len(transfer.keys),
                        transfer.ptrs,
                        event,
                    )
                    self._pending_dump_tasks.append(
                        PendingDumpTask(task, {request_id}, event)
                    )
                    self._dump_request_ids.add(request_id)
                except Exception:
                    if event:
                        self.device.destroy_event_handle(event)
                    logger.exception("Hybrid dump submit failed for %s", request_id)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if (
            not self.use_layerwise
            or self._save_complete
            or not self.has_connector_metadata()
        ):
            return
        self._seen_names.add(layer_name)
        for kind, rows in self.kv_cache_layout.rows.items():
            # A row containing multiple logical views must not be copied when
            # only one view has finished. Missing hooks are handled at end of
            # forward. State snapshots are always saved at end of forward.
            if kind == "State":
                continue
            for index, row in enumerate(rows):
                if (
                    index < len(rows) - 1
                    and row.names
                    and row.names <= self._seen_names
                ):
                    self._save_row(kind, index)

    def wait_for_save(self):
        if self._save_complete or not self.has_connector_metadata():
            return
        for layer_id in tuple(self._load_tasks):
            self._wait_layer(layer_id)
        try:
            for kind, rows in self.kv_cache_layout.rows.items():
                for index in range(len(rows)):
                    self._save_row(kind, index)
        finally:
            # Reuse the v1 task wait and event cleanup. This is not a durable
            # flush or a commit; publication remains entirely store-owned.
            self._flush_pending_dump_tasks()
            self._rank_consistency.finish_dump(self._dump_request_ids)
            self._save_complete = True

    def update_connector_output(self, connector_output):
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if meta is None:
            return
        self._rank_consistency.apply_worker_meta(meta)
        for request_id in meta.load_failed_reqs:
            self.dispatcher.requests.pop(request_id, None)

    def handle_preemptions(self, kv_connector_metadata):
        for layer_id in tuple(self._load_tasks):
            self._wait_layer(layer_id)
        super().handle_preemptions(kv_connector_metadata)
