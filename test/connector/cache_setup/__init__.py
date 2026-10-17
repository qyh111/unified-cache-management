"""The model boundary: create configuration, then construct native KV views."""


def make_config(args, rank=0):
    if args.model_config:
        from .config import make_native_config

        return make_native_config(args, rank)
    from .fixture import make_config as make_fixture_config

    return make_fixture_config(args, rank)


def construct_cache(args, rank):
    config = make_config(args, rank)
    if args.model_config:
        if args.backend == "cuda":
            from .cuda import construct_cuda_cache

            return construct_cuda_cache(args, config, rank)
        from .ascend import construct_cache as construct_ascend_cache

        return construct_ascend_cache(args, config, rank)
    from .fixture import allocate_fixture, make_native_cache_configs

    configs, specs = make_native_cache_configs(config, args.tp)
    native = configs[rank]
    allocations, views = allocate_fixture(native, specs, args.device)
    raw = {f"allocation-{i}": t for i, t in enumerate(allocations)}
    return native, views, raw, config, None
