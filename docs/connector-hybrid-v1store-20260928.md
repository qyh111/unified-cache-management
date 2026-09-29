# UCMHybridConnector v1 Pipeline 交付（2026-09-29）

## 2026-09-29 r2/r3 布局修订

当前r3版本：CPU/CUDA的FA+State采用验证边界的完整原生page，Ascend保留State语义槽位padding。每个原生 group 独立 key/记录，专用槽位策略见
[布局与生命周期清单](connector-hybrid-r2-layout-and-lifecycle.md)。旧 r1 的 FA/State 合并、尾部补行已取消；下文第六轮结果仅作为 r1 历史证据，不能当作 r2 runtime PASS。

## 基线与状态

- 开发分支：`dev_qyh_0928`，基于 fetch 后的 `origin/develop@2d24ab39`。
- 物理 view/group 解析与 scheduler 由 `c6324b8a` 的 v2 组件迁移；未迁入文件 Proxy、显式 commit 或 PCP×DCP 换算。
- 本轮新增 Hybrid，保留 Direct/HLA 作为对照与 FAWA 的继承依赖；尚未删除旧实现，也未默认切换生产入口。
- Store、native copy 实现和 FAWA 没有修改。完成/发布沿用 v1，`wait()` 不被解释为 durable flush。
- 本地仅有 Python/NumPy，无 torch、vLLM native 或 CUDA/NPU，不能宣称真实引擎、硬件 IO 或推理已通过。

## 启用与范围

在原有 UCM 启动配置中增加 `use_hybrid_connector: true`。原来的 `UCMConnector` 名称不变。

```yaml
use_hybrid_connector: true
use_layerwise: true  # 也需要用 false 单独测试 bulk
enable_event_sync: true
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: Cache|Posix
      storage_backends: /your/test/cache/path
      # 继续按机器资源配置原有 buffer/IO 参数。
```

- 单 group、多 group 均可；各 group 逻辑 `token_block_size` 必须相同，UCM key 也使用该粒度。
- 支持 FA 与 Mamba align 的区分；不要求它们共享 allocation。仅 State 的模型、不等 token 粒度、未知 spec 显式拒绝。
- FA/SWA 仍先走已有 FAWA 分流、两个 store；Hybrid 不接收 SWA。FAWA 自己的原有支持范围未扩大。
- TP1/2/4 已有部分模型的 Model-check 验收，具体以文末矩阵为准。TP 保留独立 rank key 与旧 consistency manager 路径；Hybrid 明确拒绝 PP/PCP/DCP 大于 1，相关适配与验收尚未完成，不代表已证明 v1 store 无法支持。
- 不支持 request-async load；本版 `wait_for_save` 排空已提交的 v1 任务，GPU block 不跨 step 保留。没有等待文件落盘的新协议。
- 支持 `Cache|Posix`；`Cache|Empty` 仅用于不要求恢复的存储测试。没有扩大到所有 Pipeline backend。
- 首版按初次 lookup 的 token 序列保存完整前缀；未新增 decode 持续持久化。
- MiniMax-M3 的 develop 旧实现保留；Hybrid 已完成部分无权重 Model-check，未完成真实推理验收。

## Padding 与物理地址

`hybrid/layout/` 只解释实际 view 地址、stride、segment 长度；`hybrid/store_layout.py` 独立生成固定存储模板。

每个 native group 独立保存自己的 block，在 key 中编码 group_id。不同 group 共用固定 store schema；当前要求各 group 本地模型层数相同，不再隐式补大批空行。layerwise 每层一行，bulk 将同一 group 的层模板组合为一行；单 group bulk 保持真实 segments 紧凑布局。

GLM Shared Indexer、MiniMax-M3、FA+State 和普通 attention 分别选择语义槽位策略。公共编译器只在各语义槽内做必要分段，不再把所有真实数据拍平后统一补尾。地址、stride、payload 取自 LayerView；NumPy 批量展开、v1 null 地址跳过、任务等待和发布方式不变。

## 与旧 v1 的关系

- 直接调用 `load_data(keys, indices, ptrs)` / `dump_data(keys, indices, ptrs, event)`。
- 沿用 `RankConsistencyManager` 的提交、等待、失败上报和旧事件清理。
- Cache 的空地址槽位跳过能力已存在于 native copy 实现，本轮未修改。
- Cache 完成、Posix 发布、跨 rank 数据完整性仍是原 v1 的能力边界；没有引入显式 commit、全 rank 发布事务或新的 store 服务。
- 模板文件放入单独的 `hybrid-v1-r3-…` namespace，隔离 bulk/layerwise、模型配置、包版本、设备和 cache dtype 等信息；不要将旧 Direct/HLA/v2 文件直接搬入这个目录。
- TP 各 rank 分别产生物理 key，沿用 rank0 lookup + 原 consistency manager 的策略；不假设所有 rank 共用一个文件。
- 为防不同 schema 共享 host buffer，Hybrid 的默认 `share_buffer_enable` 为 false，各 rank 的 unique ID 隔离。若显式启用共享 buffer，也不会把不同 rank 的 buffer 合并。`cache_buffer_capacity_gb` 可按需调整，一般每私有 buffer 不超过 128 GiB，须另核算多 rank 总内存。
- GC 需要真实的对齐后 block_size，由 worker 按旧机制发布；若启动顺序不能读到它，须配置准确值，首版不根据 head_size 猜测 Mamba/indexer 的大小。

## 本地检查

在仓库根目录分别执行，测试 stub 不进入生产进程：

```bash
python test/suites/Unit/test_ucm_hybrid.py
python test/suites/Unit/test_kv_cache_layout.py
python -m compileall -q ucm/integration/vllm/hybrid ucm/integration/vllm/hybrid_connector.py
```

本次 Hybrid 离线测试 18 项通过，原 KVCacheLayout 测试 33 项通过。它们使用独立 NumPy 索引校验源/目标字节与 padding canary，并通过 fake store 检查任务/事件流程；不等价于 native store 或真实推理回归。

## 从远端提交验证

源码仓库：<https://github.com/qyh111/unified-cache-management/tree/dev_qyh_0928>。
后续验证必须先推送代码，再在验证机检出明确 commit；不再用手工覆盖源码或可变 patch 作为正式验收基线。

在新目录执行（`<commit-sha>` 替换为本轮指定的完整提交号）：

```bash
git clone --branch dev_qyh_0928 --single-branch https://github.com/qyh111/unified-cache-management.git ucm-hybrid-validation
cd ucm-hybrid-validation
git checkout --detach <commit-sha>
git rev-parse HEAD
git status --porcelain
```

按项目方式分别准备 CPU/simu 与 NPU/ascend 安装，确认 native 库与源码基线兼容；不要覆盖已有验证目录和他人任务。保存 `ucm.__file__` 与 `ucm_toolkit.tools.model_check.common.__file__`，确认导入来自检出目录或由该提交构建的安装。

先运行本地检查和 `python -m unittest discover -s toolkit/tests -p 'test_model_check*.py'`；Model-check 参数和命令见 [工具说明](../toolkit/ucm_toolkit/tools/model_check/README.md)。使用安装后的 `ucm-toolkit` 命令，包没有 `python -m ucm_toolkit` 入口。

报告需记录完整 commit、工作树状态、启动命令、包版本、native 构建类型、模型配置、每 rank 的 layout、比较数量、日志和退出码。若现场修补 Python 代码，结果单独标为 dirty 调试结果；合回并推送新 commit 后重跑相关用例，才能形成正式验收。

v1 store 功能正确性作为前提，不要求运行独立 `check_hybrid_pipeline.py` 字节测试。真实推理需另外排除引擎本地 prefix cache 干扰，不能由 Model-check PASS 推导。

## CPU / NPU Model-check 路径补充

官方 vLLM 使用 CPU 包 + simu native 库代替 GPU 做无权重布局/调度/恢复验证；vLLM-Ascend 使用 NPU + ascend native 库。两者都需记录结果，CPU 通过不代表 CUDA runtime 通过。

CPU Model-check 子入口自动设置 `UCM_CPU_SIMULATION=1`，设备层使用同步 CpuDevice（event=0，store 任务仍由原 wait 排空），并放行 worker ordinal 和父类 device 选择。此模式要求 simu native 构建；不改变 store。未显式启用时不放开 CPU transfer。NPU/CUDA 的选择不受该开关影响。

CPU 与 NPU 使用独立 venv、源码构建及缓存目录，避免 .so 相互覆盖。CPU 入口及多进程 RANK、原生 MultipleOf(16) 约束已修复；两套环境均完成第六轮代表用例验证。

## 当前验证与剩余工作

本地工具 24、Hybrid 18、原布局 33 项通过，CPU/NPU 远端工具各 24 项通过。第六轮通过以下 8 个 Model-check 用例；每个用例退出码为 0，每 rank 比较数量非零且布局 rank 齐全。

| 路径 | 模型 | TP | 模式 | 每私有 buffer GiB | 每 rank payload segments |
|---|---|---|---|---|---|
| CPU | GLM5.2 | 2 | bulk | 16 | 1092 |
| CPU | MiniMax-M3 | 1 | bulk | 24 | 819 |
| CPU | Qwen3.8 | 2 | bulk | 80 | 234 |
| CPU | Qwen3.8 | 4 | bulk | 40 | 234 |
| NPU | MiniMax-M3 | 1 | bulk | 24 | 1239 |
| NPU | Qwen3.8 | 2 | bulk | 80 | 413 |
| NPU | Qwen3.8 | 2 | layerwise | 16 | 336 |
| NPU | Qwen3.8 | 4 | bulk | 40 | 413 |

CPU 引擎为 vLLM 0.29.0+cpu，NPU 为 vLLM 0.26.0+empty / Ascend 0.19.1rc2.dev1373+gcf0baa38d。独占 load buffer 数均为 512；工具暴露参数而未修改 store 默认实现。第六轮是在提交前逐个核对 26 个变更 Python 文件 SHA256 后执行，不能写成从远端 commit 检出的验证；后续按上述流程切换。

原始证据保留在工作区 `connector_v2_v1store_0928/results_remote_20260928_round6/`，未纳入本仓库；证据包 SHA256 为 `bdbe61d223cc0c21faf4dad58b0eb11c0fa8eb158a1b5e4286bb1210ce64fbd5`。

仍未完成：PP/PCP/DCP > 1、MTP、量化与真实推理、长时间多请求、抢占/取消和故障恢复验收。Eagle 暂不安排。Qwen 为未启用 W8A8 的 config 级验证；GLM NPU 结果不代表 SFA/LI C8 验收。Qwen TP1 与 Kimi bulk 仍有超过一般 128 GiB 限制的容量阻塞。已通过的 Model-check 不代表完整生产验收。
