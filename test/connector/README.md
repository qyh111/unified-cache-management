# Connector EngineCore pytest tests

Read `test_roundtrip.py` first. Its four-step cold scenario saves a prefix,
waits for native completion, closes all source ranks, opens a new EngineCore,
then loads poisoned KV and compares every rank. `runtime/session.py` owns the
EngineCore instances; `runtime/executor.py` transports RPCs with native
MessageQueue and aggregates original ModelRunnerOutputs with KVOutputAggregator.
`runtime/worker.py` uses KVConnectorModelRunnerMixin, and `runtime/state.py`
keeps native Mamba decisions/copy specifications while adapting the memcpy
launch to CPU memory. No Scheduler schedule/update loop is reimplemented.

## Run

Use **both** the local configuration and `--confcutdir`; `-c` alone does not
isolate the repository's device-dependent parent conftest.

```sh
python -m pytest -c test/connector/pytest.ini --confcutdir=test/connector \
  test/connector/test_roundtrip.py --backend ascend --device cpu --store mock \
  --runner v1 --model-config /models/glm53/config.json --tp 2 --pp 1 --mode layerwise

python -m pytest -c test/connector/pytest.ini --confcutdir=test/connector \
  test/connector --backend ascend --device cpu --tp 2 --mode bulk

python -m pytest -c test/connector/pytest.ini --confcutdir=test/connector \
  test/connector/test_roundtrip.py --backend cuda --device cpu --tp 2 \
  --model-config /models/dsv41-flash --cuda-capability 10.0 \
  --attention-backend FLASHMLA_SPARSE_DSV41

python -m pytest -c test/connector/pytest.ini --confcutdir=test/connector \
  test/connector --collect-only
```

There is no toolkit, model-check, or standalone user launcher. pytest alone
owns selection, assertions, deadlines, cleanup, and JUnit reporting. Each case
launches a new driver process before importing the engine; workers are real
spawned processes, not pytest-xdist ranks. `--output` optionally chooses a new
parent evidence directory; otherwise pytest's temporary directory is used.
Add `--junitxml=/path/results.xml` for CI.

Required engines are vLLM 0.30.0 and, for Ascend, vLLM-Ascend 0.30.0rc1.
Installed packages are the default. `--vllm-source` and `--ascend-source`
independently choose clean, exact-tag source trees and validate loaded paths.
CPU backing does not require a CPU vLLM wheel or select CpuPlatform. Ascend
imports still require torch-npu/CANN. CUDA profiles must explicitly provide
capability and attention backend; no GPU probes are guessed.

No model config selects the explicitly synthetic two-layer GQA fixture.
Model-config cases retain the actual native model config, KV specs, group IDs,
descriptors, dtype, strides, storage offsets, allocation and reshape. Weight
parameters are meta tensors. Attention/MoE/Engram computation is unexecuted;
the source writes deterministic KV bytes and sampling returns synthetic tokens.
The source runs an additional real scheduler step, so retained State checkpoints
must be confirmed by native boundary bookkeeping. State copy effects and block
zeroing are preserved. CoW requests are explicitly rejected in the core runner.

The default KV capacity ceiling is 4096 MiB per rank (`--kv-budget-mib`). The
executor probes the native configuration planner for its minimum legal block
count, reserves three times that count for this test, and checks actual native
descriptor sizes against the ceiling. EngineCore independently constructs and
distributes the final configurations. Too-small budgets fail before allocation;
there is no device profiling, compilation, warmup, server, or forward.

## Oracle, lifecycle, and limits

`data_check/comparison.py` selects payload from native views and Scheduler block
tables. Logical group/layer/component/block-position identities choose bytes;
GQA shards differ, MLA payload is replicated, and native State replication
metadata governs State shards. Expected hashes travel through test RPC only.
The connector receives unmodified opaque metadata through its public SPI.

The mock store records actual namespace/key/opaque shard identities with
lengths, digests and random-access reconstruction recipes. It retains no tensor
or payload across instances. Reads/writes are bounded to registered CPU
allocations. All reader allocation bytes are poisoned and checked before load;
only native effective payload is compared. New writes select only this step's
computed token rows, preserving history and future poison. The builder supplies
the native token-axis ABI (CUDA BHNC; Ascend MLA/SFA BNHC); indexing uses actual
strides and each spec's tokens_per_state. A passing roundtrip requires an
actual store lookup hit and load, the exact restored prefix, and positive
comparison bytes on every participating rank.

Same-instance recovery uses public `EngineCore.reset_prefix_cache()` and proves
that the next request loads externally. The direct synthetic bulk lifecycle
case explicitly enables mainline `use_request_async_load`; one rank reports
incomplete store polls across steps. Original completion results pass through
KVOutputAggregator; the test checks that Scheduler retains WAITING_FOR_REMOTE_KVS
and its block table until all ranks finish. Other routes' synchronous loads are
reported as such; that async case is skipped for those selected routes.
Store failure must surface as native request failure (or an engine exception);
byte corruption must be detected by the independent comparator. No connector
completion result is replaced, and load-error results are drained only by the
native runner lifecycle.

GLM CUDA's exact production rejection is XFAIL (expected unsupported), never a
model PASS. DeepSeek layerwise can select mainline FAWA; this does not establish
layer/compute overlap. CPU mock PASS does not establish device IO, production
Executor startup, forward numerics, real store durability, MRV2, PP, PCP/DCP,
cancellation, preemption or CoW. Device, MRV2 and PP combinations currently fail
clearly when explicitly requested. Real store is an optional extension requiring
a matching installed native simu UCM build; its mainline connector Python
files must match the checkout. It is separate from core mock evidence.

`runtime/report.py` records source SHA/status, actual package paths/versions,
model config hash, native libraries/substitutes and environment. Per-phase
layout files record actual views, allocation sizes, process IDs, poison coverage,
State copies and comparison quantities. result.json includes native boundary
records, original rank completion traces, aggregated completions and worker
exit codes. Driver timeouts terminate the case's own process group. Native SHM ownership
is recorded when created; pytest unlinks only those recorded segments after
termination and removes the case's private short IPC directory. The native
worker/scheduler geometry-file exchange uses a private case directory so its
stale-file cleanup cannot touch other engines' /dev/shm metadata. A hanging store
case proves timeout propagation and cleanup after actual load submission.
Bootstrap failure before queue creation is checked separately without an engine.
Run byte/store helper checks in an environment without a startup UCM `.pth`
import: the mock guard deliberately rejects an already imported native logger.

## Sources and attribution

The branch starts at develop `7164b73119b21ccb48d4876d324b3afa08c6624e`.
The following flesher0813 commits are preserved in order with `cherry-pick -x`:

- `8b888d77f61d066eec49965095a4ab1014891149`: original no-weight CUDA/Ascend model check.
- `cce530e1e9a1ca37fa1892904b4e9ecffe620962`: bounded block planning.
- `e722dd22f739e3c582d32ba6293a41f935798cd3`: backend block-size update before KV specs.
- `6d19146ac83433fb0968f8abfe1fa86be4ed89b1`: standard native KV adaptation.

Construction and metadata-only model adaptations incorporate subsequent 0.30
fixes from `test/connector-offline@e5f641f333aca64ec740cefb291f8e13c9b99228`:
`cache_setup/{config,cuda,ascend,fixture}.py`, independent `data_check/`, engine
source bootstrap and provenance. The previous scheduler/worker callback loop,
scenario and launcher are not copied. New code is under test only; the examples
introduced by these four commits are removed after migration. develop's existing
unrelated examples, toolkit, production connector and v2 implementation are not
changed. Original commit authorship and source trailers remain in Git history.
