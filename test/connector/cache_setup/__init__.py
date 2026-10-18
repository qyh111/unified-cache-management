"""Configuration and native worker KV construction (no scheduler)."""


def make_config(args, rank=0):
    if args.model_config:
        from .config import make_native_config

        return make_native_config(args, rank)
    from .fixture import make_config as fixture_config

    return fixture_config(args, rank)


def prepare(args, config, rank):
    if not args.model_config:
        from .fixture import fixture_specs

        return None, fixture_specs(config)
    if args.backend == "cuda":
        from .cuda import prepare_runner
    else:
        from .ascend import prepare_runner
    return prepare_runner(args, config, rank)


def allocate(args, runner, native, specs):
    if runner is None:
        from .fixture import allocate_fixture

        raw_list, views = allocate_fixture(native, specs, args.device)
        raw = {f"allocation-{i}": t for i, t in enumerate(raw_list)}
        kernels = None
        token_axis = 2  # explicitly declared synthetic B,KV,N,H,C
    else:
        if args.backend == "cuda":
            from .cuda import allocate_runner

            token_axis = 2  # native allocate_kv_cache: B,H,N,C (v0.30)
        else:
            from .ascend import allocate_runner

            token_axis = 1  # native Ascend MLA/SFA get_kv_cache_shape: B,N,H,C
        views, raw, kernels = allocate_runner(runner, native)
    from ..data_check.comparison import components

    axes = {}
    persistent = {
        name
        for gid in native.prefix_cacheable_group_ids
        for name in native.kv_cache_groups[gid].layer_names
    }
    for name, value in views.items():
        spec = specs[name]
        if name not in persistent or spec.tokens_per_state < 0:
            continue
        axes[name] = token_axis
        for tensor in components(value):
            factor, remainder = divmod(tensor.shape[0], native.num_blocks)
            if (
                remainder
                or tensor.shape[token_axis] * factor * spec.tokens_per_state
                != spec.block_size
            ):
                raise ValueError(f"Native token-axis contract differs for {name}")
    return views, raw, kernels, axes


def allocation_upper_bound(args, native):
    sizes = [descriptor.size for descriptor in native.kv_cache_tensors]
    if args.model_config and args.backend == "cuda":
        # v0.30 allocate_kv_cache allocates one shared buf and aliases every
        # descriptor into it. Summing descriptor.size counts that buf repeatedly.
        if len(set(sizes)) != 1:
            raise ValueError("Native CUDA shared allocation descriptors differ")
        return sizes[0]
    return sum(sizes)
