"""Real Scheduler phases; connector metadata is opaque to the test runtime."""

import time
from dataclasses import asdict, dataclass

from ..cache_setup.config import resolve_native_length
from .environment import require


def model_output(req_ids, kv_output):
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req: i for i, req in enumerate(req_ids)},
        sampled_token_ids=[[7] for _ in req_ids],
        kv_connector_output=kv_output,
    )


def aggregate(aggregator, output, results):
    req_ids = list(output.num_scheduled_tokens)
    return aggregator.aggregate([model_output(req_ids, item[0]) for item in results])


def make_request(name, tokens, hash_size):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256_cbor
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256_cbor)
    return Request(
        request_id=name,
        prompt_token_ids=list(range(10, tokens + 10)),
        sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_size, sha256_cbor),
    )


@dataclass(frozen=True)
class SavedPrefix:
    """Only engine block IDs, source hashes and worker provenance survive."""

    blocks: tuple
    hashes: list
    workers: list


class SchedulerSession:
    """Save one prefix, then restore it through a fresh Scheduler and workers."""

    def __init__(self, args, config, workers):
        from vllm.v1.core.kv_cache_utils import (
            generate_scheduler_kv_cache_config,
            resolve_kv_cache_block_sizes,
        )

        self.args, self.config, self.configs = args, config, workers.configs
        config.cache_config.block_size = workers.info[0]["resolved_block_size"]
        if args.model_config:
            resolve_native_length(args, config)
        config.cache_config.kv_cache_layout = self.configs[0].kv_cache_layout
        config.cache_config.num_gpu_blocks = self.configs[0].num_blocks
        self.native = generate_scheduler_kv_cache_config(self.configs)
        self.block_size, self.hash_size = resolve_kv_cache_block_sizes(
            self.native, config
        )
        # Retain the source scheduler/store until the entire roundtrip ends,
        # matching its previous lifetime while workers restart independently.
        self.source_scheduler = None

    def _new_scheduler(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.structured_output import StructuredOutputManager

        return Scheduler(
            self.config,
            self.native,
            StructuredOutputManager(self.config),
            block_size=self.block_size,
            hash_block_size=self.hash_size,
            log_stats=False,
        )

    def save_prefix(self, workers):
        from vllm.config import set_current_vllm_config
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator

        with set_current_vllm_config(self.config):
            scheduler = self.source_scheduler = self._new_scheduler()
            aggregator = KVOutputAggregator.from_connector(
                scheduler.connector, self.args.tp
            )
            source = make_request("source", self.args.tokens, self.hash_size)
            scheduler.add_request(source)
            output = scheduler.schedule()
            require(
                output.num_scheduled_tokens.get("source") == self.args.tokens,
                "Source prefix was not fully scheduled",
            )
            blocks = scheduler.kv_cache_manager.get_block_ids("source")
            results = workers.step(output, "source", blocks)
            hashes = [r[3] for r in results]
            scheduler.update_from_output(output, aggregate(aggregator, output, results))
            self._drain(scheduler, aggregator, workers)
            require(source.is_finished(), "Source request did not finish")
            return SavedPrefix(blocks, hashes, workers.info)

    def load_and_compare(self, workers, saved):
        from vllm.config import set_current_vllm_config
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator

        require(
            not {i["pid"] for i in saved.workers} & {i["pid"] for i in workers.info},
            "Source worker was reused",
        )
        require(
            [asdict(c) for c in workers.configs] == [asdict(c) for c in self.configs],
            "Cold reader native configs changed",
        )
        with set_current_vllm_config(self.config):
            scheduler = self._new_scheduler()  # empty native prefix cache
            aggregator = KVOutputAggregator.from_connector(
                scheduler.connector, self.args.tp
            )
            target = make_request("target", self.args.tokens + 2, self.hash_size)
            hit_tokens = self._wait_for_external_prefix(scheduler, target)
            scheduler.add_request(target)
            output = scheduler.schedule()
            blocks = scheduler.kv_cache_manager.get_block_ids("target")
            results = workers.step(output, "target", blocks, hit_tokens, saved.hashes)
            merged = aggregate(aggregator, output, results)
            require(
                not merged.kv_connector_output.invalid_block_ids,
                "Load reported invalid blocks",
            )
            require(
                all(r[1] > 0 and r[2] > 0 for r in results),
                "A rank compared no KV bytes",
            )
            scheduler.update_from_output(output, merged)
            self._drain(scheduler, aggregator, workers)
            require(target.is_finished(), "Target request did not finish")
            return self._result(workers, saved, blocks, hit_tokens, results)

    def _wait_for_external_prefix(self, scheduler, target):
        deadline = time.monotonic() + self.args.timeout
        while True:
            hit_tokens, _ = scheduler.connector.get_num_new_matched_tokens(target, 0)
            if hit_tokens == self.args.tokens or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        require(
            hit_tokens == self.args.tokens,
            f"External lookup restored {hit_tokens}/{self.args.tokens} tokens",
        )
        return hit_tokens

    @staticmethod
    def _drain(scheduler, aggregator, workers):
        output = scheduler.schedule()
        scheduler.update_from_output(
            output, aggregate(aggregator, output, workers.step(output))
        )

    def _result(self, workers, saved, blocks, hit_tokens, results):
        return {
            "status": "PASS",
            "store": self.args.store,
            "backend": self.args.backend,
            "device": self.args.device,
            "fixture": (
                "native-model-cache"
                if self.args.model_config
                else "synthetic-full-attention"
            ),
            "mode": self.args.mode,
            "tp": self.args.tp,
            "tokens": self.args.tokens,
            "source_workers": saved.workers,
            "reader_workers": workers.info,
            "source_blocks": saved.blocks,
            "target_blocks": blocks,
            "external_hit_tokens": hit_tokens,
            "compared_segments_by_rank": [r[1] for r in results],
            "compared_bytes_by_rank": [r[2] for r in results],
            "control_processes": "driver + TP workers; launcher, resource tracker and mock Manager are support processes",
        }
