# 无卡 UCMConnector 黑盒测试

从公开 vLLM SPI 驱动主线 `ucm.integration.vllm.ucm_connector.UCMConnector`，
最终检查每个 rank 加载后的 KV 字节 hash 是否等于保存前的 hash。
不启动服务、EngineCore、Executor 或 forward，不读取 connector 的私有
request/dispatch/layout/task 状态，也不根据它生成预期结果。

默认 `--store mock`：只使用 checkout 中的 UCM Python 代码，**不需要安装或编译 UCM**。
`--store real` 切换到已安装的 simu UCM Cache|Posix，执行真实文件读写。
这里测试 connector 的接线和数据恢复；store 自身的一致性、耐久性另行测试。

## 先读主流程

先读 **[scenario.py](scenario.py) 的 `run_roundtrip()`**。这里仅保留四步：
准备环境和模型配置 → 保存源 KV → 源 worker 退出 → 新 worker 加载并比较。
日常阅读不需要展开模型构造、无卡适配或报告代码。

```text
connector/
├── run.py                 # 唯一命令行入口，参数及启动环境
├── scenario.py            # 黑盒测试流程：从这里开始读
├── runtime/               # 真实 Scheduler、TP 进程与环境支撑
│   ├── scheduler.py       # save_prefix / load_and_compare 两个阶段
│   ├── workers.py         # worker 生命周期及公开 connector SPI 回调
│   ├── environment.py     # 引擎来源、版本、平台和 CPU device 适配
│   └── report.py          # 输出目录、结果及源码/包来源记录
├── cache_setup/           # 只构造原生 KV，不执行 forward
│   ├── __init__.py        # make_config / construct_cache 统一入口
│   ├── config.py          # 模型配置及公开 store 注入配置
│   ├── ascend.py          # Ascend 原生 KV 构造
│   ├── cuda.py            # CUDA 平台及原生 KV 构造
│   └── fixture.py         # 不传 model_config 时的合成 FA 示例
├── data_check/            # 独立数据校验与 store 依赖
│   ├── comparison.py      # 从原生 KV 生成源 hash、poison 和比较
│   ├── memory.py          # 确定性字节流、已注册 CPU 内存和重建信息
│   ├── mock_store.py      # 默认 hash/recipe store
│   └── real_store.py      # 可选已安装 simu store
└── tests/                 # 上述辅助模块的自测，不是模型用例
    ├── test_cli.py
    ├── test_environment.py
    └── test_data.py
```

按问题继续展开即可：

| 想了解什么 | 继续读 |
| --- | --- |
| Scheduler 如何保存和冷加载 | [runtime/scheduler.py](runtime/scheduler.py) |
| worker 调用了哪些公开 SPI | [runtime/workers.py](runtime/workers.py) |
| 前后数据如何独立比较 | [data_check/comparison.py](data_check/comparison.py) |
| key/shard/hash 如何存取 | [data_check/mock_store.py](data_check/mock_store.py) |
| 具体 backend 的真实 tensor 布局如何获得 | [cache_setup/](cache_setup/) |
| 无卡运行、已安装包/源码选择 | [runtime/environment.py](runtime/environment.py) |

依赖方向为 `scenario → runtime/cache_setup/data_check`。cache_setup 不导入 Scheduler 或比较器；
比较器不读取 store 记录或 UCM 私有计划。测试边界和数据判定保持不变。

## 一次测试的流程

1. 创建真实 Scheduler 和 TP workers。具体模型使用原生模型类、KV spec、
   分组、allocation 和 reshape 方法；KV backing 位于 CPU。
2. Scheduler 分配源请求 block。fixture 独立填充确定性数据，并直接按原生
   tensor 的 shape/stride/storage_offset 和引擎 block table 保存预期 hash。
3. 公开回调执行保存，返回值经过真实 `KVOutputAggregator` 汇总，再送回 Scheduler。
4. **源 workers 正常退出并销毁所有 KV 内存**。跨进程只保留 hash、重建信息和引擎配置。
5. 新建 reader workers 和空缓存 Scheduler；通过公开命中接口确认外部前缀命中。
   目标 tensor 整体填充 `0xff`，然后调用真实 connector 的加载接口。
6. 每次 `wait_for_layer_load` 后，直接比较该层原生 KV payload 的 hash。
   每个 rank 必须有非零比较量，失败返回非零退出码。

源和目标是不同进程的 allocation；各自 Scheduler 的数字 block ID 可以相同。
独立的预期结果不使用 UCM 的 key、shard、地址数组或 dispatch，避免根据待测实现
反推预期值。null block 和原生非 prefix-cacheable scratch 不属于恢复 payload。
单次 State align case 比较可复用边界和运行槽；不代表多轮 State/forward 数值验证。

TP2 同时有一个 scheduler driver 和两个 worker。源、reader 两组 workers 顺序运行，
不复用进程。launcher、multiprocessing resource tracker 和共享 mock ledger Manager
属于辅助进程。具体模型构造使用真实 Gloo 控制通信；没有 Executor RPC。

## mock store 如何只保存 hash

记录身份为 **`(store namespace, 实际 key, 实际 shard_index)`**。
shard_index 是 store SPI 的不透明编号；layerwise 可以是物理 row，不能假定等于层号。

源数据来自可随机寻址的确定性字节流，包含 allocation 名称、rank 和物理偏移的区别。
保存时 mock 读取 connector **实际提交的地址和长度**，计算 hash，并验证它符合 fixture
已有的重建信息。记录只有长度、hash、种子和字节范围，不保存 tensor 或 payload 副本。
加载时按该记录重建字节，校验记录 hash，再写入 connector **实际提交的目标地址**。
独立 oracle 将目标内容与源快照 hash 比较。错 key、错 shard、错目标、漏加载或错层，
不能通过重新根据本次调用生成“预期数据”来掩盖。

MLA 的复制 KV 和 TP 分片 State 分别准备数据；GQA 的 rank payload 彼此不同。
mock 只允许访问已注册的原生 CPU allocation；空指针/零长度 padding 按真实 store
scatter/gather 规则跳过，不能作为有效目标数据的来源。
它在 wait/check 时完成操作，用于 connector 功能验证，不模拟真实 IO 延迟和并发时序。

## 环境与参数

- Linux，Python >= 3.10，PyTorch、NumPy、Transformers、wrapt 及对应引擎导入依赖。
- 只支持 **vLLM 0.30.0**，NPU 分支还需要 **vLLM-Ascend 0.30.0rc1**。
  允许对应 release 的本地 build suffix；不兼容旧版本。
- 默认使用已安装引擎；`--vllm-source` / `--ascend-source` 可独立覆盖对应包。
  显式源码必须是精确 release tag，tracked files 干净；无自动 fallback。
- `--backend cuda|npu` 选择原生平台、patch 和布局；`--device cpu` 只决定 KV tensor
  放置位置。**没有原生 CPU attention 布局**，目前也没有真实 CUDA/NPU allocation 模式。
- Ascend 仍需 torch-npu/CANN 等 Python 导入所依赖的库和匹配 `_build_info.py`；
  无卡不代表可以省略引擎的软件依赖。可以直接使用已有 Ascend 容器。
- mock 模式替换 UCM logger/metrics 的 native 依赖，并阻止加载 UCM `.so`；
  不替换 connector 的 hashing、lookup 决策、Layout、地址生成或完成结果。
- real 模式需要安装以 `RUNTIME_ENVIRONMENT=simu` 构建的 UCM；CPU 指针不能交给
  CUDA/NPU 构建的 store。容量默认 32 GiB，仅当实际 shard 需要满足 Cache 的
  1024-buffer 最小值时自动提高，队列容纳无 forward 间隔的逐层提交。

每次传入一个**新的** `--output` 目录。只读取本地模型材料，离线模式禁止下载。

```sh
# 已安装 Ascend 引擎，默认纯 Python mock store，不需要卡或 UCM 编译：
python test/connector/run.py --backend npu --device cpu \
  --model-config /models/glm53/config.json --tp 2 --mode layerwise \
  --output /tmp/ucm-glm-blackbox

python test/connector/run.py --backend npu --device cpu \
  --model-config /models/dsv41-flash/config.json --tp 2 --mode bulk \
  --output /tmp/ucm-dsv-blackbox

# 指定源码是可选的；没有源路径时使用已安装包：
python test/connector/run.py --backend npu \
  --vllm-source /vllm-workspace/vllm \
  --ascend-source /vllm-workspace/vllm-ascend \
  --tp 2 --output /tmp/ucm-fa-blackbox

# 可选真实 store；需已安装 simu UCM：
python test/connector/run.py --backend npu --store real \
  --tp 2 --mode layerwise --output /tmp/ucm-real-store

# CUDA 无卡布局：显式给出 capability 和 backend，不探测 GPU：
python test/connector/run.py --backend cuda --device cpu \
  --vllm-source /sources/vllm --cuda-capability 10.0 \
  --attention-backend FLASHMLA_SPARSE_DSV41 \
  --model-config /models/dsv41-flash/config.json --tp 2 \
  --output /tmp/ucm-cuda-layout

# 辅助自测：入口/环境检查不需要引擎；数据检查需要 CPU torch
# 缺少 torch 时仅数据自测会明确 skip，不影响 --help 和入口检查：
python -m unittest discover -s test/connector/tests -p 'test_*.py'
PYTHONPATH=test:. ENABLE_UCM_PATCH=0 UCM_CAPTURE_VLLM_LOG=0 \
  python -m unittest connector.tests.test_data -v
```

## 原生构造和覆盖边界

模型参数在 meta 上构造，不读权重。KV-only runner 调用原生 spec/allocator/reshape，
保留真实 shape、dtype、stride、offset 和 alias；不初始化执行缓冲、计算 metadata 或
forward。未执行的 MoE、Engram 计算/权重分配等被显式替代，误执行会失败。
CUDA 还替代硬件探测及未执行计算的导入入口，源模式缺失的注册扩展使用空导入占位，
不提供 kernel 实现；安装包模式保留真实扩展/ABI 错误。Ascend 应用真实原生 patch。

`--tokens` 可指定模型请求长度，默认依据原生 block 对齐。目标在源前缀后追加两个 token，
保留引擎 recompute 所需空间。本测试保留原生约束，不能为通过而修改模型参数/backend。

`--mode layerwise` 只表示配置选择。当前 develop 的 DeepSeek V4.1 优先选择 FAWA
connector，因此该配置不证明执行了逐层流水。GLM 的逐 row 存取也不证明与计算重叠。
GLM CUDA 原生布局可构造，但当前主线 UCM 明确拒绝，必须作为不支持报告。

结果记录在 `result.json`、`source/` 和 `reader/`，包括代码 SHA/dirty、实际包路径、
模型材料 hash、原生配置/视图、每 rank 比较量、源/目标 PID、加载的 native 模块及退出码。
失败不算模型 PASS；mock 通过不替代 real store、设备 DMA、kernel、权重推理、
数值 State、多轮抢占/取消/CoW、PP/CP/DP 或跨 rank 事务的验收。
