# DSpark / UCM NaN 与接收率诊断包

目标组合：vllm-ascend 0.25.1rc、UCM develop 2026-07-31 最后提交 `26464d0ddba21175ae91a73a25b544941431e0b3`、`DeepSeek-V4-Flash-0731-w8a8`、TP8、DSpark 7、MRV1。

## 0. 从 qyh 分支直接获取

```bash
git clone -b feature/dspark-nan-diagnostic-20261116 \
  https://github.com/qyh111/unified-cache-management.git
cd unified-cache-management
export UCM_HOME=$PWD
export DIAG_ROOT=$PWD/test/diagnostics/dspark
chmod +x "$DIAG_ROOT"/*.sh "$DIAG_ROOT"/*.py
"$DIAG_ROOT/apply_diagnostics.sh"
```

这里 `UCM_HOME` 是仓库根目录，`DIAG_ROOT` 是诊断文件目录，两者不要设成同一路径。若直接使用本文的 `/home/qyh/dspark-0731-debug-20261116` 布局，则保持脚本默认值即可。

## 1. 这个包做什么

- `ascend-nan-diagnostic.patch`：在 target hidden、DSpark combine、draft context KV、draft logits、target logits、target argmax 和 draft token IDs 处增加 `[DSPARK_DIAG]` 日志。
- `ucm-nan-diagnostic.patch`：在 FAWA 外部 KV load 完成后及 dump 提交前，按 request/group/layer/block 检查实际 HBM view 的 NaN/Inf。
- `run_native_diagnostic.sh`：原生 `vllm serve`，不使用 dummy 或 torchrun。
- `run_ucm_diagnostic.sh`：同一模型和请求配置，加 UCM `Cache|Posix`。
- `request_probe.py`：发送固定长 prompt，保存生成文本及 Prometheus 接收率差值。
- `summarize_diagnostic.py`：汇总非零 NaN/Inf 和首个 target/draft ID。

所有检查只在 `VLLM_ASCEND_DSPARK_DIAG=1` 时启用。**启用后会有 NPU 同步和全 tensor finite 扫描，只用于诊断，不用于性能测试或生产。**

## 2. 应用补丁

默认路径：

- vllm-ascend：`/vllm-workspace/vllm-ascend`
- UCM：`/home/qyh/dspark-0731-debug-20261116/ucm-26464d0`

```bash
cd /home/qyh/dspark-0731-debug-20261116
chmod +x diagnostics/*.sh
./diagnostics/apply_diagnostics.sh
```

`apply_diagnostics.sh` 会先做 `git apply --check`，不会强行应用。若公司环境源码版本不同，先核对差异，不要强套 patch。恢复：

```bash
./diagnostics/revert_diagnostics.sh
```

若 UCM 尚未编译：

```bash
docker exec <container> bash /home/qyh/dspark-0731-debug-20261116/diagnostics/build_ucm.sh
```

`build_ucm.sh` 使用 `PLATFORM=ascend`，对 `/home/qyh/dspark-0731-debug-20261116/ucm-26464d0` 做 editable install。公司环境可设置 `UCM_HOME`、`VLLM_ASCEND_HOME`、`DIAG_ROOT` 覆盖默认路径。

## 3. 推荐验证顺序

### A. 无 UCM 基线

```bash
./diagnostics/run_native_diagnostic.sh
# 服务 ready 后：
python3 diagnostics/request_probe.py \
  --url http://127.0.0.1:18080 \
  --out evidence/native \
  --label cold
```

`GPU_MEM` 默认 `0.70`，仅为短请求诊断预留；若机器空闲且要复现生产配置，可设 `GPU_MEM=0.9`。`MODEL`、`PORT` 也可覆盖。

先看冷请求是否已异常。**全新空存储的首次请求没有外部 KV 可加载，不能归因 Posix。**

### B. UCM 冷请求

换新的、确认不存在的 Posix 目录；不要把不同权重/配置写到同一目录。默认目录：

`/home/qyh/dspark-0731-debug-20261116/store-posix`

```bash
./diagnostics/run_ucm_diagnostic.sh diagnostics/ucm_posix.yaml
python3 diagnostics/request_probe.py \
  --url http://127.0.0.1:18080 \
  --out evidence/ucm \
  --label cold
```

### C. 同进程 HBM 命中

再次发送同一 prompt，但不 reset：

```bash
python3 diagnostics/request_probe.py \
  --url http://127.0.0.1:18080 \
  --out evidence/ucm \
  --label hbm-hit
```

### D. 清 HBM 后外部命中

调用 `/reset_prefix_cache` 后再发同一 prompt：

```bash
python3 diagnostics/request_probe.py \
  --url http://127.0.0.1:18080 \
  --out evidence/ucm \
  --label external-hit \
  --reset
```

关键是 C 与 D 的差异：如果 A/B 正常，而 D 才异常，才优先查 UCM 已持久化的错误字节、layout、旧记录或 load 生命周期。

可选把 `ucm_cache_empty.yaml` 替换 `ucm_posix.yaml`，区分 connector/layout 与磁盘 IO。不要复用旧 Cache/Posix 记录做结论。

## 4. 日志怎么读

```bash
python3 diagnostics/summarize_diagnostic.py /path/to/server.log
# 全部诊断行：
python3 diagnostics/summarize_diagnostic.py /path/to/server.log --full
```

### 4.1 先确认是否真有 NaN/Inf

非零示例：

```text
[DSPARK_DIAG] target.draft_input_hidden_states ... nan=12 +inf=0 -inf=0 ...
[DSPARK_DIAG] ucm.load.request=... ... nan=4 ...
[DSPARK_DIAG] draft.context_kv.layer=43 ... nan=2 ...
```

只有 `nan=0 +inf=0 -inf=0` 才表示该 view 没有 NaN/Inf。**没有 NaN 不代表 KV 字节正确**；stale、错位、错误 block 都可能是有限值。

### 4.2 按层级定位

- `target.draft_input_hidden_states`：target aux 进入 DSpark 前的数值。
- `draft.combined_hidden_states`：`main_norm(main_proj(...))` 后。
- `draft.context_kv.layer=43/44/45`：三个 DSpark draft 层写入前的 shared KV。
- `draft.raw_logits` / `draft.token_ids`：draft 输出及 proposal。
- `sampler.target_logits` / `sampler.target_argmax`：target verification。
- `ucm.load...`：UCM load 完成后、forward 前的实际 HBM block。
- `ucm.dump...`：dump 提交前的实际 HBM block。

如果 `ucm.load` 已异常，说明加载结果或原记录有问题；如果 load 正常但后续 forward/dump 异常，优先查当前计算，而非 Posix。

### 4.3 接收率必须是真样本

看 `request_probe.py` 输出：

- `drafts > 0 && draft_tokens > 0 && accepted == 0` 才是真的 0 接收；
- `drafts == 0` 或 `draft_tokens == 0` 是没有样本，不是 0；
- `mean_acceptance_length` 接近 `1.00` 时，才与 0 接收一致。

Greedy 下首个 draft ID 不等于 target argmax 就会拒绝该位置，且后续不再接收。**NaN 可以导致接收率 0，但代码没有“遇 NaN 自动清零”的分支，不能把 0 直接等同 NaN。**

对照：

```text
sampler.target_argmax ids=[...]
sampler.draft_token_ids ids=[...]
```

若首个 ID 相同但 metrics 为 0，应改查 output/scheduler/metrics，而不是继续查 NaN。

## 5. UCM 侧已知但需区分的事项

0731 UCM 对 DSpark 的三个 draft 层按普通 WA 注册，但 scheduler GC 估算只加 1 层：

- scheduler WA：`6,627,328 B`
- 实际 WA：`6,889,472 B`
- 少算：`262,144 B`

这会使 GC 偏晚/容量超配，**不是直接的 NaN 证明**。启动日志出现这两个值时可记录，但不要把它单独当作接收率 0 的根因。

UCM hash 不包含完整权重 revision/draft 层数/KV layout 身份，因此务必为不同权重、不同引擎修复或不同配置使用全新后端目录，避免 false-hit。

## 6. 当前验证边界

补丁已做 Python 语法检查及 `git apply --check --reverse` 校验。它不会改变默认数值路径，但启用诊断会同步并增加开销。当前本地/共享环境尚未完成一次有效服务端到端验证，因此不要把包本身当成已验证根因修复；它是用于公司环境采集判定证据的工具。
