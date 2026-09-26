# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Unified group-schema connector: HLA's scheduling face, per-group
physical addressing.

``UCMGroupSchemaConnector`` inherits the proven scheduler flow of
``UCMHybridLinearAttentionConnector`` (per-group hash chains, two-stage
lookup, ``group_plan`` dispatch slices) and swaps the physical layer:
``GroupSchemaLayout`` compiles the ported connector-v2 pipeline
(``parse_kv_cache_config`` -> group layouts -> record templates ->
``StoreSchema``), and load/dump resolve **per group** over the full slot
table with the other groups' slots ghosted. The shared-page assumption
(one flat block-id list addressing every group's segments) exits here:
each group's independent block-id chain resolves its own slots.

Store submission shape is unchanged -- one flat ``(key, slot)`` grid per
request, ghost slots carrying ``ptr=0`` at their declared width, which
the store skips by the established ghost convention.

Opt-in via launch_config ``use_group_schema_layout``; bulk only (the
layerwise per-layer dispatch integration is a later step).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from ucm.integration.vllm.hla_connector import (
    UCMHybridLinearAttentionConnector,
)
from ucm.integration.vllm.layout import GroupSchemaLayout
from ucm.integration.vllm.ucm_connector import UCMConnectorMetadata
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.store.ucmstore_v1 import Task

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = init_logger(__name__)


class UCMGroupSchemaConnector(UCMHybridLinearAttentionConnector):
    """Bulk connector over the group-schema layout (per-group addressing)."""

    def _create_kv_cache_layout(
        self, kv_caches: dict
    ) -> GroupSchemaLayout:
        return GroupSchemaLayout(
            kv_caches,
            self.launch_config,
            self._vllm_config,
            self._kv_cache_config,
        )

    def _scope_slice_keys(self, sl, is_dump: bool) -> Optional[list[bytes]]:
        """Rank-scope one slice's store keys, or ``None`` when dropped.

        Mirrors ``_scope_blocks``/``_mla_split_scope`` semantics per slice:
        on MLA models the full-attention blocks share the rank-0 hash and
        are dumped by rank 0 only; everything else (KDA, mamba state) is
        re-hashed per rank on non-rank-0.
        """
        is_rank0 = self.tp_rank % self.tp_size == 0
        if self.is_mla and not sl.is_state:
            if is_dump and not is_rank0:
                return None
            return list(sl.ucm_block_ids)
        if is_rank0:
            return list(sl.ucm_block_ids)
        return [self.request_hasher(b) for b in sl.ucm_block_ids]

    def _resolve_slices(self, slices):
        """Resolve slices into one submission grid over the slot table.

        Returns ``(keys, ptrs)``: *keys* are the scoped store keys in
        slice order; *ptrs* is the ``(n_keys, n_slots)`` grid where each
        key's row resolves its owning group's slots and ghosts the rest.
        """
        schema = self.kv_cache_layout.store_schema
        keys: list[bytes] = []
        grids: list[np.ndarray] = []
        for sl in slices:
            scoped = self._scope_slice_keys(sl, is_dump=False)
            if not scoped:
                continue
            _, ptrs, _ = self.kv_cache_layout.resolve_group(
                sl.group_id, sl.vllm_block_ids
            )
            keys.extend(scoped)
            grids.append(np.asarray(ptrs, dtype=np.uint64).reshape(len(scoped), -1))
        if not grids:
            return keys, None
        return keys, np.concatenate(grids, axis=0)

    def _resolve_dump_slices(self, slices):
        """Dump-side variant: returns ``(rank0_keys, keys, ptrs)``.

        *rank0_keys* are the un-scoped hashes per surviving key (the
        tracker set), *keys* the scoped store keys.
        """
        schema = self.kv_cache_layout.store_schema
        rank0_keys: list[bytes] = []
        keys: list[bytes] = []
        grids: list[np.ndarray] = []
        for sl in slices:
            scoped = self._scope_slice_keys(sl, is_dump=True)
            if not scoped:
                continue
            _, ptrs, _ = self.kv_cache_layout.resolve_group(
                sl.group_id, sl.vllm_block_ids
            )
            rank0_keys.extend(sl.ucm_block_ids)
            keys.extend(scoped)
            grids.append(np.asarray(ptrs, dtype=np.uint64).reshape(len(scoped), -1))
        if not grids:
            return rank0_keys, keys, None
        return rank0_keys, keys, np.concatenate(grids, axis=0)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Bulk load over per-group schema resolution."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        request_to_task: dict[str, Task] = {}
        is_load = False
        num_loaded_block = 0
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        request_to_load_blocks: dict[str, int] = {}
        # Ensure do_mamba_copy_block (from preprocess_mamba, compute stream)
        # has completed before submitting load DMA (store stream).
        self.device.synchronize()
        for request_id, request in metadata.request_meta.items():
            slices = tuple(getattr(request, "load_slices", ()) or ())
            if not slices:
                continue
            is_load = True
            scoped_keys, ptrs = self._resolve_slices(slices)
            if not scoped_keys:
                continue
            num_loaded_block += len(scoped_keys)
            num_loaded_request += 1
            rank0_keys = [
                block_id for sl in slices for block_id in sl.ucm_block_ids
            ]
            try:
                shard_indexs = [0] * len(scoped_keys)
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: rank0_keys},
                    scoped_keys,
                    shard_indexs,
                    ptrs,
                )
                request_to_task[request_id] = task
                request_to_load_blocks[request_id] = len(scoped_keys)
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. "
                    f"{type(e).__name__}: {e}"
                )
                self._record_load_error(
                    "connector_load_submit_errors_total",
                    request.load_block_ids[1] + request.dump_block_ids[1],
                )
                self._connector_worker_meta.mark_failed(request_id)
                num_loaded_block -= len(scoped_keys)

        for request_id, task in request_to_task.items():
            try:
                self._rank_consistency.wait_load(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait load task error. "
                    f"{type(e).__name__}: {e}"
                )
                self._record_load_error(
                    "connector_load_wait_errors_total",
                    metadata.request_meta[request_id].load_block_ids[1]
                    + metadata.request_meta[request_id].dump_block_ids[1],
                )
                self._connector_worker_meta.mark_failed(request_id)
                num_loaded_block -= request_to_load_blocks.get(request_id, 0)

        if is_load:
            load_end_time = time.perf_counter() * 1000
            load_duration_ms = load_end_time - load_start_time
            load_bytes = num_loaded_block * self.block_data_size
            load_speed = load_bytes / max(load_duration_ms, 1) / 1024 / 1024
            ucmmetrics.update_stats(
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_duration_ms,
                    "load_speed": load_speed,
                    "load_bytes_total": load_bytes,
                }
            )

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        block_ids_by_request: dict[str, set[bytes]] = {}
        num_saved_block = 0
        all_keys: list[bytes] = []
        grids: list[np.ndarray] = []
        for request_id, request in metadata.request_meta.items():
            slices = tuple(getattr(request, "dump_slices", ()) or ())
            if not slices:
                continue
            rank0_keys, scoped_keys, ptrs = self._resolve_dump_slices(slices)
            if not scoped_keys:
                continue
            block_ids_by_request[request_id] = set(rank0_keys)
            num_saved_block += len(scoped_keys)
            all_keys.extend(scoped_keys)
            grids.append(ptrs)

        if not all_keys:
            return

        event_handle = 0
        try:
            total_ptrs = np.concatenate(grids, axis=0)
            shard_indexs = [0] * len(all_keys)
            event_handle = self._get_dump_event_handle()
            save_start_time = time.perf_counter() * 1000
            task = self._rank_consistency.submit_dump(
                self.store,
                block_ids_by_request,
                all_keys,
                shard_indexs,
                total_ptrs,
                event_handle,
            )
        except Exception as e:
            logger.error(f"dump kv cache failed. {type(e).__name__}: {e}")
            if self.enable_event_sync and event_handle and self.device is not None:
                self.device.destroy_event_handle(event_handle)
            self._rank_consistency.finish_dump(set(block_ids_by_request))
            return

        try:
            self._rank_consistency.wait_dump(task)
            save_end_time = time.perf_counter() * 1000
        except Exception as e:
            logger.error_limit(
                f"wait for dump kv cache failed. {type(e).__name__}: {e}"
            )
            self._rank_consistency.finish_dump(set(block_ids_by_request))
            return
        finally:
            if self.enable_event_sync and event_handle and self.device is not None:
                self.device.destroy_event_handle(event_handle)

        self._rank_consistency.finish_dump(set(block_ids_by_request))
        save_bytes = num_saved_block * self.block_data_size
        ucmmetrics.update_stats(
            {
                "save_duration": save_end_time - save_start_time,
                "save_bytes_total": save_bytes,
            }
        )
