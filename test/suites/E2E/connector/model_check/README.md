# MRV2 Parallel Model-Check (UCM v1 & v2)

Card-less MRV2 verification harness for testing UCM connector parallel topologies
(`TP`, `PP`, `PCP`, `DCP`, and their combinations) across **both v1 and v2**
connectors without circularly depending on UCM's internal layout logic.

## How it works

1. **MRV2 Meta Build + Native CPU KV Allocation** (`host_runtime.py`, `native_cache.py`, `parallel.py`):
   Launches `TP × PCP × PP` local Gloo processes via `torchrun`, initializes real
   vLLM `TP`/`PP`/`PCP`/`DCP` process groups, builds each PP stage on `meta`,
   and allocates real MRV2 KV tensor views on CPU.
2. **Unified Mock Byte Store** (`mock_store.py`):
   Shared cross-process file-backed byte store with two thin adapters:
   - `--connector v1`: intercepts `UcmConnectorFactoryV1.create_connector` (preserves
     `storage_backends` directory isolation, layerwise shard commit, and ghost slots).
   - `--connector v2`: intercepts `StoreSchedulerClient` / `StoreWorkerClient`.
   Every pointer copy is bounds-checked against registered native KV buffers, and
   cross-rank writes are audited for conflicting or duplicate writes.
3. **Engine-Native Poison + Roundtrip Oracle** (`verifier.py`, `run.py`):
   - **Cold Write**: v1 starts the KV pool poisoned, writes replica-consistent
     bytes to the current Scheduler step's native block-table locations, and
     snapshots the prefix before later steps can reclaim its blocks. Retained
     State checkpoints use the Scheduler's native boundary hand-offs, with
     distinct bytes for running and checkpoint pages when the slots differ.
     The harness drives layer callbacks in model layer order around a simulated
     forward.
   - **Negative Isolation**: mutates token 0 and verifies zero false external hits.
   - **Poison + Hot Reload**: resets local HBM prefix cache, poisons all KV buffers
     with `0xA5`, reloads a shared-prefix request from the mock store, and checks
     every engine-native expected slice (`prefix_cacheable_group_ids`, `block_tables`,
     and native `kv_caches` views) byte-for-byte.
4. **v1 parallel fault checks**: The GLM-5.3 PP2 and DCP2 matrix cases hide a
   committed key after a successful reload and verify that the hit stops before
   it. They also check native guard pages outside the hot request's allocation.
   Direct MLA DCP cases check each Store key's writer against its Scheduler
   shard position and the writer's native DCP rank.

## Examples (Ascend A2/A3 server)

### 1. Provision official A2/A3 ModelScope quantized metadata (no weights)

```bash
python test/suites/E2E/connector/model_check/fetch_models.py \
  --out /home/qyh/ucm-a2-quant-models
```

This fetches only lightweight metadata (`config.json`, `quant_model_description.json`,
tokenizer/processor configs, and custom `*.py` modeling files) for the official
`Eco-Tech/*` A2/A3 checkpoints documented in `vllm-ascend/docs/source/tutorials/models/`,
and writes `<out>/manifest.json`. `test_parallel_model_check.py` automatically adds
`--quantization ascend` whenever `quant_model_description.json` is present.

### 2. Run checks

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_ASCEND_ARCH=Ascend910B3

# Single custom parallel check (v1, PP2 x TP2 x PCP2 x DCP2)
python test/suites/E2E/connector/model_check/run.py \
  --connector v1 \
  --platform ascend-sim --ascend-soc A2 \
  --quantization ascend --kv-cache-dtype int8 --indexer-kv-dtype int8 \
  --trust-remote-code \
  --model /home/qyh/ucm-a2-quant-models/ascend/glm-5.3 \
  --tp 2 --pp 2 --pcp 2 --dcp 2 \
  --block-size 128 --cp-kv-cache-interleave-size 128 \
  --tokens 4096 --max-num-batched-tokens 2048

# Run pytest parallel matrix (default: Connector v1, 28 roundtrip + 2 v1 reject,
# covering layerwise, whole-block request-async with --load-delay-polls,
# and MultiConnector [MockPDHandshakeConnector, UCMConnector] PP=1/PP=2).
# Set ENABLE_UCM_PATCH=1 to also activate UCM runtime monkey-patches (e.g. the
# MultiConnector PP handshake routing patch) when running directly from source.
# Each roundtrip case uses --auto-align-chunks 4 so padded Mamba/KDA/FAWA block
# sizes automatically size prompt length and per-step batch budget to 4 aligned
# chunks (guaranteeing a real State checkpoint at 75%).
export UCM_V2_MODEL_ROOT=/home/qyh/ucm-a2-quant-models
export UCM_MODEL_CHECK_PLATFORM=ascend
export UCM_MODEL_CHECK_CONNECTOR=v1
python -m pytest -c test/pytest.ini --confcutdir=test/suites/E2E/connector -o addopts= -vs \
  test/suites/E2E/connector/test_parallel_model_check.py
```

For an asynchronous v1 Dump path, add `--dump-delay-polls 64` to a GLM-5.3
`run.py` command. The harness requires a pending Dump when it schedules a
second prefill request and checks that the source bytes remain intact until the
Store consumes them. It also resolves the pending source addresses back to
native block IDs and checks that aborting the first request leaves those blocks
referenced and that the second request allocates none of them.

For request-async v1 Load verification, pass `--no-use-layerwise` (or run a
FAWA model, which always loads asynchronously) with `--load-delay-polls 4`. The
harness asserts the hot request stays in `WAITING_FOR_REMOTE_KVS` across steps,
verifies its destination blocks stay referenced and untouched while an
overlapping prefill request executes, and simulates MRV2's align-mode Mamba
state pre-copy (`checkpoint -> running`) on the promotion step before running
the byte-for-byte oracle.

To enable UCM runtime monkey-patches on demand when running from a source
checkout without `ucm_patch.pth`, pass `--enable-ucm-patch` to `run.py` or set
`ENABLE_UCM_PATCH=1` in the environment.
