"""Optional installed simu Cache|Posix backend, selected through store SPI."""

from ucm.store.pipeline.connector import UcmPipelineStore


class RealStore(UcmPipelineStore):
    def __init__(self, config):
        config = dict(config)
        if int(config.get("device_id", -1)) >= 0:
            shard = int(config["shard_size"])
            config["cache_buffer_capacity_gb"] = max(
                32, (1024 * shard + (1 << 30) - 1) // (1 << 30)
            )
        super().__init__(config)
