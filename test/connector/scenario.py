"""Read this first: save KV, destroy the source, cold-load and compare hashes.

Runtime drives public SPI, cache_setup builds native KV views, and data_check
owns independent expectations/store dependencies. No UCM private state is read.
"""

import multiprocessing as mp

from .cache_setup import make_config
from .cache_setup.fixture import TOKENS
from .runtime.environment import initialize_runtime
from .runtime.report import main
from .runtime.scheduler import SchedulerSession
from .runtime.workers import Workers


def run_roundtrip(args):
    initialize_runtime(args)
    config = make_config(args)
    if not args.model_config:
        args.tokens = TOKENS

    with mp.get_context("spawn").Manager() as manager:
        ledger, lock = manager.dict(), manager.RLock()
        if args.store == "mock":
            from .data_check.mock_store import bind, summarize

            bind(None, ledger, lock)

        # Successful scope exit stops and joins every source rank.
        with Workers(args, "source", ledger, lock) as source:
            scheduler = SchedulerSession(args, config, source)
            saved = scheduler.save_prefix(source)

        # Source memory no longer exists. Reader KV is poisoned before load;
        # each layer is compared to the independent hashes in saved.
        with Workers(args, "reader", ledger, lock) as reader:
            result = scheduler.load_and_compare(reader, saved)

        result["store_records"] = summarize(ledger) if args.store == "mock" else {}
        return result


if __name__ == "__main__":
    raise SystemExit(main(run_roundtrip))
