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
   - **Cold Write**: fills native KV tensors with replica-consistent deterministic
     bytes and runs a full chunked-prefill Dump through the real vLLM `Scheduler`
     and `KVOutputAggregator`.
   - **Negative Isolation**: mutates token 0 and verifies zero false external hits.
   - **Poison + Hot Reload**: resets local HBM prefix cache, poisons all KV buffers
     with `0xA5`, reloads a shared-prefix request from the mock store, and checks
     every engine-native expected slice (`prefix_cacheable_group_ids`, `block_tables`,
     and native `kv_caches` views) byte-for-byte.

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

# Run pytest parallel matrix against the provisioned A2/A3 ModelScope root
export UCM_V2_MODEL_ROOT=/home/qyh/ucm-a2-quant-models
export UCM_MODEL_CHECK_PLATFORM=ascend
export UCM_MODEL_CHECK_CONNECTOR=v1
python -m pytest -c test/pytest.ini --confcutdir=test/suites/E2E/connector -o addopts= -vs \
  test/suites/E2E/connector/test_parallel_model_check.py
```
