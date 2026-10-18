"""Optional installed simu Cache|Posix backend, selected through store SPI."""

from ucm.store.pipeline.connector import UcmPipelineStore


CONTROL = LOCK = None


def bind(control, lock):
    global CONTROL, LOCK
    CONTROL, LOCK = control, lock


def observe(name, count):
    with LOCK:
        CONTROL[name] = CONTROL.get(name, 0) + int(count)


class RealStore(UcmPipelineStore):
    def __init__(self, config):
        config = dict(config)
        if int(config.get("device_id", -1)) >= 0:
            shard = int(config["shard_size"])
            config["cache_buffer_capacity_gb"] = max(
                32, (1024 * shard + (1 << 30) - 1) // (1 << 30)
            )
        super().__init__(config)

    def lookup(self, block_ids):
        result = super().lookup(block_ids)
        observe("lookup_hits", sum(result))
        return result

    def lookup_on_prefix(self, block_ids):
        result = super().lookup_on_prefix(block_ids)
        observe("lookup_hits", result + 1)
        return result

    def lookup_on_reverse(self, block_ids):
        result = super().lookup_on_reverse(block_ids)
        observe("lookup_hits", int(result >= 0))
        return result

    def load_data(self, block_ids, shard_index, dst_addr):
        result = super().load_data(block_ids, shard_index, dst_addr)
        observe("loads", len(block_ids))
        return result
