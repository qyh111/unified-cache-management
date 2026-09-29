"""Storage identity shared by scheduler and worker, independent of view specs."""

import hashlib
import json


def storage_namespace(
    spec,
    *,
    model,
    device,
    cache_dtype,
    model_dtype,
    tp,
    layerwise,
    versions,
    additional_config=None,
    quantization=None,
):
    # generate_scheduler_kv_cache_config collapses a per-layer UniformType map
    # to one representative spec. Never hash its per-layer repr/physical sizes.
    identity = {
        "model": model,
        "device": device,
        "cache_dtype": str(cache_dtype),
        "model_dtype": str(model_dtype),
        "tp": tp,
        "versions": versions,
        "additional_config": additional_config,
        "quantization": quantization,
        "groups": [
            {
                "id": group.group_id,
                "tokens": group.token_block_size,
                "state": group.is_state_snapshot,
                "layers": sorted(layer.layer_name for layer in group.layers),
            }
            for group in spec.groups
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"hybrid-v1-r4-{digest}-b{spec.ucm_cache_block_size}-lw{int(layerwise)}"
