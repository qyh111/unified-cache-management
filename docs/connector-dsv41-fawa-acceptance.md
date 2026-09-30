# DeepSeek V4.1 Flash FAWA 首版验收

基线：6bf673a0；日期：2026-09-30。本文描述本轮实现，不是运行时通过报告。

## 实现范围

- 按 `hf_text_config.model_type=deepseek_v41_text`（或 deepseek_v41）进入独立 schema；V4 的 C4/C128 分支保留。
- FA 调度块128，ratio 1/2，各层物理行数从 spec 核对；SWA window128，物理块必须整除128。
- CPU/CUDA-sim 混合页，NPU 不同行数的主 KV、int8 indexer 和 fp16 scale 分别建地址列。支持捕获的 dense page 内容及块间/层间 padding，不将引擎的 padded page 字节当作 store payload。
- scheduler/worker 共用逐注册 owner 的字节 schema，替代 V4 层数常量；worker 比对每层行数、组件字节数和最终4KB对齐大小，不一致立即拒绝。
- FA/WA 仍为两个 Cache|Posix store，新增 `dsv41-r1-<schema摘要>` 后缀隔离旧文件；不修改 store 接口或完成发布语义。
- ring 保留原生 group ID，作为零 tail WA 占位，不读取/存储其地址。仅支持当前采集的 CPU容量8、NPU容量32。每个 ring 必须匹配同源层 C2 主缓存。
- 不支持 speculative/MTP、PP/PCP/DCP>1，明确拒绝。TP留给本轮运行时验收。

## 环恢复依据与边界

核查源码版本：vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、vllm-ascend `b64b4d714484feaa6ca71edc99b98318fe6d2f0d`。

- CPU `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py`：request program 只有 `(position+1)%2==0` 才读 predecessor ring；其余完整 pair 从本 chunk 原始输入读取。
- NPU `vllm_ascend/ops/triton/compressor/compressor_triton.py`：`_pool_kernel` 只有 `residual=start_pos-group_idx*2>0` 才从历史环拼接；`compressor_from_projected` 的池化先于环写入。
- 外部命中必须恢复至128的整数倍，故初始 start_pos 为偶数、开放组为空。后续奇数 chunk 起点需要的前驱来自本请求恢复后的新计算。此论证不依赖 capacity 是否整除128。
- 完整 prompt 命中不能沿用旧 FAWA “减1 token”的策略，那会产生奇数恢复起点。新分支查询更早的完整且真实存在的 WA 快照；没有这样的快照就 miss 重算，不能仅把最后一个 hit 数值减小。
- 这不是 speculative 回退证明，首版对此明确拒绝。

## SWA 快照与生命周期

长 prefill 中旧 SWA 块可能已被回收，V4.1 强制 chunk-wise，仅发布本轮最后一个完整128边界的 WA。非对齐结尾或窗口包含 null block 时不发布 WA，但完整 FA 前缀仍可保存；lookup 需要 FA连续前缀和真实存在的 WA边界同时满足。`dump_wa` 元数据供 worker 和 model-check 使用。

`wait_for_save` 返回前等待该模型的 dump 完成，避免后续 native block_pool 复用源块。使用现有 wait 和错误处理，不增加 store commit。

## 本地单测

在仓库根执行：

```sh
python test/suites/Unit/test_dsv41_layout.py
python test/suites/Unit/test_ucm_hybrid.py
python test/suites/Unit/test_glm53_layout.py
python -m unittest discover -s toolkit/tests -p 'test_model_check*.py'
```

DSV4.1 测试用真实 FAWA 类源码和 NumPy tensor stand-in，隔离 torch/vllm 导入；不是 native scheduler 或 NPU 算子验收。覆盖 schema、每行 scale、混合行数/stride地址、非法布局、ring配对、完整命中恢复边界、WA快照条件及 wait 阻塞。独立 C2 依赖 oracle 从投毒 ring 开始，跨奇偶 chunk 验证历史行来自恢复后的输入。

## 交给验证 agent

1. fetch `qyh/dev_qyh_0928`，记录本轮最终完整SHA、clean状态、实际加载路径，不使用旧的6bf673a0结果冒充新实现。
2. 复用 r6 的模型材料 revision `dba1be0a40aa45a94ad051997016db3960a90277`、0.30环境和 command.sh。完整权重不需要。
3. CPU `UCM_MODEL_CHECK_DSV41_FP8_MOCK=1`；NPU `UCM_MODEL_CHECK_LOAD_FORMAT=dummy`。保持原生 spec/group/shape，不改引擎分组。使用 FAWA，**不加 --hybrid**。
4. 基础参数：block128、tokens8192、无speculative、TP1；CPU保持r6实际的KV dtype，NPU保持auto/bf16。buffer先16GB，按需调整且一般不超过128GB。
5. 先跑本地四套件，加装有torch/vllm的 `test_hma_4d_layout.py`。CPU/NPU TP1各跑layerwise/bulk，成功后TP2，再TP4。NPU先查占用、避开6/7，不停服务或重置设备。
6. 从日志确认选中UCMFAWAConnector，hash128、两store新namespace、ring未进入tensor_size_list。CPU完整40层采集的对齐大小应FA=229376B、WA=2990080B；NPU应FA=372736B、WA=5242880B。大小以实际注册spec/视图为准，若不符保存差异，不能强行改常量通过。
7. source_prompt_tokens 应产生完整128边界的source，target增加后缀。核查实际source长度与load结束位置；每rank比较计数>0、exit0、无traceback，保存dump/load原生block IDs不同的证据。
8. 单列负例：相同长度且仅最后边界有WA时应安全miss；source结尾非128对齐不应发布WA；窗口null不发布WA；speculative/MTP、PP/PCP/DCP>1显式拒绝。不把负例拒绝计作模型PASS。
9. 真实CPU/CUDA与NPU内核可用时，单列C2恢复依赖验证：先正常prefill至偶数边界，保存持久KV/indexer/SWA；新请求恢复且ring填毒值，从偶数位置继续，分块长度1/3/5等，和不中断基线比较。此项是模型环语义验证，不是store字节专项；算子不可用就列未验，不能mock读写环来宣称数值通过。
10. 回归DSV4 Flash CPU原有FAWA双模式、GLM5.3 CPU TP2双模式(144/rank)、Qwen3.8 CPU TP2双模式(128/rank)。只有schema变更才更新namespace版本，不能为了跑通沿用旧store文件。

每case交付 command.sh、exit-code.txt、run.log；保存引擎/模型/UCM身份、tensor_size_list摘要和逐rank比较结果。把“已实现、model-check通过、环数值验证、未运行/阻塞”分开报告。遇到问题先提交最小证据，不自行去掉安全门。
