"""Unified scheduler-side lookup, request state, and dispatch planning."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from .ucm_kv_cache import UCMKVCacheGroupInfo, UCMKVCacheSpec
from .ucm_proxy import UCMProxyAdapter

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class RequestHasher:
    """MD5 hasher compatible with the existing connector namespace format."""

    def __init__(self, vllm_config: "VllmConfig", rank_id: int | None) -> None:
        speculative = getattr(vllm_config, "speculative_config", None)
        spec_info = ""
        if speculative is not None:
            method = getattr(speculative, "method", "") or ""
            tokens = getattr(speculative, "num_speculative_tokens", 0)
            spec_info = f":{method}:{tokens}"
        additional = getattr(vllm_config, "additional_config", None) or {}
        sparse = (
            f":sfa_c8={int(bool(additional.get('enable_sparse_sfa_c8', False)))}"
            f":li_c8={int(bool(additional.get('enable_sparse_li_c8', False)))}"
        )
        model_config = vllm_config.model_config
        model_name = model_config.model.rstrip("/").split("/")[-1]
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        meta = (
            f"{model_name}:{tp_size}:{model_config.dtype}:"
            f"{rank_id}{spec_info}{sparse}"
        )
        self.meta_bytes = meta.encode("utf-8")

    def __call__(self, value: object) -> bytes:
        payload = (
            value
            if isinstance(value, bytes)
            else pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        )
        return hashlib.md5(self.meta_bytes + payload).digest()


@dataclass(frozen=True)
class UCMLookupResult:
    external_hit_tokens: int
    group_ucm_block_ids: tuple[tuple[bytes, ...], ...]


@dataclass
class RequestState:
    hbm_hit_tokens: int = 0
    external_hit_tokens: int = 0
    num_token_ids: int = 0
    token_processed: int = 0
    group_ucm_block_ids: tuple[tuple[bytes, ...], ...] = ()
    group_vllm_block_ids: tuple[list[int], ...] = ()
    load_pending: bool = False


@dataclass(frozen=True)
class UCMGroupDispatchPlan:
    hash_group: Literal["FA", "WA", "State"]
    keys: tuple[bytes, ...]
    token_start: int
    token_end: int
    # Window block ids per participating group, in dispatch_routes()
    # order -- the static window shape lives in the spec (see
    # ``group_window_shape``), so only the ids travel.
    windows: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class RequestDispatchMeta:
    request_id: str
    load_plans: tuple[UCMGroupDispatchPlan, ...] = ()
    dump_plans: tuple[UCMGroupDispatchPlan, ...] = ()


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    requests: dict[str, RequestDispatchMeta] = field(default_factory=dict)
    preempted_req_ids: set[str] = field(default_factory=set)
    finished_req_ids: set[str] = field(default_factory=set)


def _token_ids(request: "Request") -> tuple[int, ...]:
    # int() normalizes numpy scalars so the hash chain pickles stably.
    return tuple(int(value) for value in request.all_token_ids)


_KEY_TYPE_BITS: Mapping[str, int] = {"FA": 0, "WA": 1, "State": 2}


def _key_tag(chain: str, tp_rank: int = 0, pp_rank: int = 0) -> bytes:
    """The 2-byte suffix of a UCM key, big-endian bit layout:

    type(2) group(4) tp_rank(4) pp_rank(4) reserved(2).  Every group of a
    chain shares one record, so no group bits are set today; the layout
    keeps them for a future per-group split.  Ranks default to the
    logical rank-0 key namespace the scheduler hashes in.
    """

    value = (
        (_KEY_TYPE_BITS[chain] << 14)
        | ((tp_rank & 0b1111) << 6)
        | ((pp_rank & 0b1111) << 2)
    )
    return value.to_bytes(2, "big")


class UCMDispatcher:
    """Scheduler-side lookup, request state, and dispatch planning.

    ``lookup`` probes the external cache and records the per-request
    snapshot; ``build_from_scheduler_output`` then maintains block tables
    from vLLM's SchedulerOutput and produces pointer-free plans.
    """

    def __init__(
        self,
        kv_cache_spec: UCMKVCacheSpec,
        proxy: UCMProxyAdapter,
        request_hasher: RequestHasher,
        base_seed: bytes,
        *,
        load_threshold_tokens: int = 0,
        recompute_tokens: int = 1,
        tp_rank: int = 0,
        pp_rank: int = 0,
    ) -> None:
        self.spec = kv_cache_spec
        self.proxy = proxy
        self.hasher = request_hasher
        self.base_seed = base_seed
        self.load_threshold_tokens = max(int(load_threshold_tokens), 0)
        self.recompute_tokens = max(int(recompute_tokens), 0)
        self.requests: dict[str, RequestState] = {}
        # The per-kind routing table (key kind -> participating groups),
        # built once: FA, then the tail-storing WA groups, then State.
        self._routes = kv_cache_spec.dispatch_routes()
        self._chain_tags = tuple(
            (label, _key_tag(label, tp_rank, pp_rank)) for label, _ in self._routes
        )

    def _chain(
        self, token_ids: Sequence[int], block_size: int, parent: bytes
    ) -> tuple[bytes, ...]:
        result: list[bytes] = []
        for start in range(0, len(token_ids), block_size):
            block = tuple(token_ids[start : start + block_size])
            if len(block) != block_size:
                break
            parent = self.hasher((parent, block))
            result.append(parent)
        return tuple(result)

    def _chain_keys(
        self, token_ids: Sequence[int]
    ) -> tuple[tuple[bytes, ...], ...]:
        """Per-chain keys: one shared hash chain, each chain's tag appended.

        The chain runs over ucm_cache_block_size token blocks from the
        base seed; a key is the chain value's first 14 bytes plus the
        chain's 2-byte tag, so the FA and boundary keys at one boundary
        differ only in their tag.
        """

        values = self._chain(
            token_ids, self.spec.ucm_cache_block_size, self.base_seed
        )
        return tuple(
            tuple(value[:14] + tag for value in values)
            for _label, tag in self._chain_tags
        )

    def lookup(self, request: "Request", num_computed_tokens: int) -> UCMLookupResult:
        """Probe the external cache and record the request snapshot."""
        token_ids = _token_ids(request)
        result = self._lookup(token_ids, num_computed_tokens)
        if result.external_hit_tokens <= self.load_threshold_tokens:
            result = UCMLookupResult(0, result.group_ucm_block_ids)
        self.requests[str(request.request_id)] = RequestState(
            hbm_hit_tokens=num_computed_tokens,
            external_hit_tokens=result.external_hit_tokens,
            num_token_ids=len(token_ids),
            token_processed=num_computed_tokens + result.external_hit_tokens,
            group_ucm_block_ids=result.group_ucm_block_ids,
            group_vllm_block_ids=tuple([] for _ in self.spec.groups),
            load_pending=result.external_hit_tokens > 0,
        )
        return result

    def _lookup(
        self, token_ids: Sequence[int], hbm: int
    ) -> UCMLookupResult:
        """FA prefix restore; WA/State additionally need their boundary.

        All chains share one hash chain at ucm_cache_block_size, so every
        chain has a key at the same boundary.  The FA chain is a prefix
        requirement; the WA (window tail) and State (mamba snapshot)
        chains are boundary records -- restoring requires a complete FA
        prefix up to some boundary and that boundary's tail or snapshot.
        Each boundary chain is reverse-scanned to its latest hit; the
        restore boundary is the earliest of those (the latest boundary
        where every chain hits), never past the FA prefix.
        """

        ucm_block_size = self.spec.ucm_cache_block_size
        group_ucm_block_ids = self._chain_keys(token_ids)
        # Every chain scans up to the last complete block below the
        # recompute margin; the margin's block is left to recompute (v1
        # semantics -- a full hit still recomputes recompute_tokens).
        last = max(len(token_ids) - self.recompute_tokens, 0) // ucm_block_size
        first = hbm // ucm_block_size
        fa_end = hbm
        for (label, _tag), keys in zip(self._chain_tags, group_ucm_block_ids):
            if label == "FA":
                hits = self.proxy.lookup_on_prefix(keys[first:last])
                if hits >= 0:
                    fa_end = max((first + hits + 1) * ucm_block_size, hbm)
                break
        restore_end = fa_end
        for (label, _tag), keys in zip(self._chain_tags, group_ucm_block_ids):
            if label == "FA":
                continue
            hits = self.proxy.lookup_on_reverse(keys[first : fa_end // ucm_block_size])
            if hits < 0:
                restore_end = hbm  # no boundary record: nothing may restore
                break
            restore_end = min(restore_end, (first + hits + 1) * ucm_block_size)
        return UCMLookupResult(max(restore_end - hbm, 0), group_ucm_block_ids)


    def update_blocks(
        self,
        request_id: str,
        group_block_ids: Sequence[Sequence[int]],
        *,
        append: bool,
    ) -> None:
        state = self.requests[request_id]
        if len(group_block_ids) != len(self.spec.groups):
            raise ValueError("group block table count does not match KV cache groups")
        if append:
            for destination, source in zip(state.group_vllm_block_ids, group_block_ids):
                destination.extend(int(value) for value in source)
        else:
            state.group_vllm_block_ids = tuple(
                [int(value) for value in source] for source in group_block_ids
            )

    def build_metadata(
        self,
        scheduled_tokens: Mapping[str, int],
        *,
        preempted_req_ids: Sequence[str] = (),
        finished_req_ids: Sequence[str] = (),
    ) -> UCMConnectorMetadata:
        metadata = UCMConnectorMetadata(
            preempted_req_ids=set(preempted_req_ids),
            finished_req_ids=set(finished_req_ids),
        )
        for request_id, num_scheduled in scheduled_tokens.items():
            state = self.requests.get(str(request_id))
            if state is None:
                continue
            metadata.requests[str(request_id)] = self._request_meta(
                str(request_id), state, int(num_scheduled)
            )
        for request_id in (*preempted_req_ids, *finished_req_ids):
            self.requests.pop(str(request_id), None)
        return metadata

    def build_from_scheduler_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> UCMConnectorMetadata:
        """Consume the vLLM 0.26 SchedulerOutput shape.

        New and resumed block tables replace the snapshot; ordinary cached
        allocations append only their newly allocated blocks.
        """

        for request in scheduler_output.scheduled_new_reqs:
            request_id = str(request.req_id)
            if request_id in self.requests:
                self.update_blocks(
                    request_id,
                    request.block_ids,
                    append=False,
                )

        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id_value in enumerate(cached.req_ids):
            request_id = str(request_id_value)
            if request_id not in self.requests:
                continue
            incoming = cached.new_block_ids[index]
            resumed = request_id in cached.resumed_req_ids
            if incoming is not None:
                self.update_blocks(request_id, incoming, append=not resumed)

        return self.build_metadata(
            scheduler_output.num_scheduled_tokens,
            preempted_req_ids=tuple(scheduler_output.preempted_req_ids or ()),
            finished_req_ids=tuple(scheduler_output.finished_req_ids),
        )

    def _request_meta(
        self, request_id: str, state: RequestState, scheduled_tokens: int
    ) -> RequestDispatchMeta:
        step_end = min(state.token_processed + scheduled_tokens, state.num_token_ids)
        should_load = state.load_pending and scheduled_tokens > 0
        load_end = state.hbm_hit_tokens + state.external_hit_tokens
        load_start = state.hbm_hit_tokens if should_load else load_end
        dump_start = state.token_processed
        dump_end = step_end
        load = self._plans(state, load_start, load_end, is_dump=False)
        if should_load:
            state.load_pending = False
        dump = self._plans(state, dump_start, dump_end, is_dump=True)
        state.token_processed = step_end
        return RequestDispatchMeta(request_id, load, dump)

    def _plans(
        self, state: RequestState, token_start: int, token_end: int, *, is_dump: bool
    ) -> tuple[UCMGroupDispatchPlan, ...]:
        plans: list[UCMGroupDispatchPlan] = []
        if token_end <= token_start:
            return ()
        ucm_block_size = self.spec.ucm_cache_block_size
        for route_index, (hash_group, physical_groups) in enumerate(self._routes):
            keys_available = state.group_ucm_block_ids[route_index]
            if hash_group in ("WA", "State"):
                # One boundary record per plan.  A dump keeps the newest
                # boundary the step completed -- a step may straddle
                # boundaries without ending on one, so completion is
                # "token_end passed it" (the step range, which starts
                # where the previous step ended, keeps each boundary from
                # being recorded twice); a step completing no new boundary
                # stores nothing.  A load restores the boundary load_end
                # ends on (always cache-aligned).
                if is_dump:
                    if token_end // ucm_block_size <= token_start // ucm_block_size:
                        continue
                elif token_end % ucm_block_size:
                    continue
                boundary = token_end // ucm_block_size
                start = max(boundary - 1, 0)
                end = boundary
            else:
                start = token_start // ucm_block_size
                end = token_end // ucm_block_size
                if end <= start:
                    continue
            if hash_group == "WA":
                # A tail window is only worth storing once it is
                # complete: an early boundary shorter than the chain's
                # largest tail would clamp the window head to token 0 --
                # the request's own prefix, which a restore recomputes
                # anyway.  Skipping keeps every dumped key's record
                # complete, so reverse lookup never selects a partial
                # one (the load branch above cannot select one either).
                if end * ucm_block_size < max(
                    group.tail_tokens or 0 for group in physical_groups
                ):
                    continue
            plans.append(
                UCMGroupDispatchPlan(
                    hash_group,
                    tuple(keys_available[start:end]),
                    start * ucm_block_size,
                    end * ucm_block_size,
                    tuple(
                        self._group_blocks(hash_group, group, state, start, end)
                        for group in physical_groups
                    ),
                )
            )
        return tuple(plans)

    def _group_blocks(
        self,
        hash_group: Literal["FA", "WA", "State"],
        group: UCMKVCacheGroupInfo,
        state: RequestState,
        start: int,
        end: int,
    ) -> np.ndarray:
        """One group's window block ids for the plan's keys, vectorized.

        The static half of the window (block count, sub-span shape) lives
        in ``group_window_shape``; only these ids travel in the plan.
        FA keys each span one whole unit: whole blocks when the unit is
        the larger side, the containing block per key otherwise.  A WA
        plan's single boundary key keeps the tail window's blocks (the
        _plans gate guarantees the window is complete and the parse-time
        alignment check guarantees the boundary is block-aligned).  A
        State plan's key keeps the boundary's last block.
        """

        ucm_block_size = self.spec.ucm_cache_block_size
        table = state.group_vllm_block_ids[group.group_id]
        token_block = group.token_block_size
        if hash_group == "State":
            last = (end * ucm_block_size - 1) // token_block
            return np.asarray(table[last : last + 1], dtype=np.uint64)
        if hash_group == "WA":
            assert group.tail_tokens  # WA chains only carry tail-storing groups
            boundary = end * ucm_block_size
            first = (boundary - group.tail_tokens) // token_block
            return np.asarray(
                table[first : boundary // token_block], dtype=np.uint64
            )
        if ucm_block_size % token_block == 0:
            # Whole blocks: one table slice carries every key's span.
            return np.asarray(
                table[
                    start * ucm_block_size // token_block : end
                    * ucm_block_size
                    // token_block
                ],
                dtype=np.uint64,
            )
        # Several keys share one block: the containing block per key (the
        # table is short here -- token_block spans at least two units).
        table_ids = np.asarray(table, dtype=np.uint64)
        ordinals = np.arange(start, end, dtype=np.uint64) * ucm_block_size
        return table_ids[ordinals // token_block]
