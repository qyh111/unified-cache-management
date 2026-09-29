# Ring 配对修复验收与 DeepSeek V4.1 Flash 适配输入

日期：2026-09-30。修复基线：66f7f4de。最终测试必须记录实际提交 SHA，不能用基线的日志替代。

## 本轮实现与本地验证

Qwen raw_key_cache 逐个匹配同一完整 indexer 前缀的 compressed_key_cache；拒绝缺失、重复和非 MLA 配对。其他层的缓存不能作为本层环缓冲可丢弃的依据，压缩比仅从本层配对读取。

新增跨层错误配对、同名非 MLA 两项拒绝测试。本地 Hybrid 44 项、GLM5.3 5 项通过。没有修改 store、完成发布语义和 padding 策略。

## 请验证 agent 执行

1. 从远端获取包含上述修复的提交，记录 SHA、工作区状态、引擎版本、ucm 实际加载路径。不要验证未推送的源码副本。
2. 运行 `python test/suites/Unit/test_ucm_hybrid.py`、`python test/suites/Unit/test_glm53_layout.py` 和 `python -m unittest discover -s toolkit/tests -p 'test_model_check*.py'`。
3. 复用 r5 Qwen Next CPU 原生配置及构造 mock，确认正确配对通过门控，之后仍在尚未实现的专用布局处明确拒绝。保存完整 traceback，不能将这个预期拒绝计作模型 PASS。
4. GLM5.3 CPU TP2、Qwen3.8 CPU TP2 各运行 layerwise/bulk，比较每 rank 144/128 segments。此次改动不影响既有 NPU 布局，若实际代码变更扩大再补 NPU 回归。
5. 每 case 保存 command.sh、exit-code.txt、run.log；报告区分本次提交验证与沿用历史证据。

## DeepSeek V4.1 Flash：新适配项，尚未宣称支持

模型配置来源：https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json 。下载时固定模型仓 revision，保存原始 config 和 SHA256。

2026-09-30 读取的配置：外层 `deepseek_v41`，文本 `deepseek_v41_text`，40 层、SWA 128，压缩比列表包含 0/2/1，存在独立的 `kv_source_layer_ids` 与 `index_source_layer_ids`，以及 3 个 nextn 层。不能直接按 V4 的 4/128 压缩布局或按每个消费者层复制缓存。

已直接读取远端 nightly 容器的 `vllm/models/deepseek_v41/compressor.py` 与 `attention.py`：

- compressor 当前实现 ratio 1/2；ratio 2 的 CircularBufferSpec 保存 float32 的 kv/score，state_dim=1024。
- 环容量为 `max(8, next_power_of_two(num_speculative_tokens + 2))`，绑定视图由 `[B,1,N,C]` squeeze 为 `[B,N,C]`。
- ratio 1 无 compressor ring；ratio 0 的具体缓存角色须从真实注册 spec 核实，不能拿 compressor 的 ratio 1/2 限制否定整个模型。
- attention 引用 SWA cache、共享源层与 indexer，indexer 支持 FP8/FP4 字节布局。因此初步按 FAWA 双 store 评估，而不是强塞 Hybrid 单 store。

### 第一阶段：捕获真实布局，不改原生分组

在 0.30 CPU/NPU 分别记录 config、engine SHA、registered layers、spec 类型、token_block_size、tokens_per_state、dtype、shape、stride、storage offset、共享源层映射、group 列表和 descriptor。先无 speculative，再加实际引擎支持的 MTP 配置。TP1 成功后 TP2/4。

允许 mock 计算 factory/可用性探针，必须保持原生 spec/shape/group，不运行 forward。不要把 FP8 权重量化等同于 KV dtype。NPU 若构造或分组阻塞，原样保存 traceback，不能为了通过复制 CUDA 布局。

### 第二阶段：确定适配补丁

- 确认 FAWA 对 CircularBufferSpec 的分组、查找和 block table 使用方式。请求私有环不能直接按 token 前缀块寻址。
- 证明 ratio-2 完整组恢复是否可省略 ring；逐源层检查配对、dump/load 边界及 speculative 回退。未证明前保持拒绝，不能套用 Qwen 门控。
- 共享 KV/indexer 按真实 owner 去重，消费者别名不能重复 dump；核对不同压缩比的调度块到物理行转换。
- 仅在原生布局输入与生命周期约束明确后实现 V4.1 专用适配，补独立地址 oracle 单测，再交 model-check。

### 验收边界

已有 V4 Flash PASS 不代表 V4.1 PASS；布局通过也不代表请求私有环、跨轮复用或 MTP 生命周期通过。最终报告分别列出两类验收。

Cache|Posix 及 v1 store 保持不变，buffer 按需且一般不超过128GB。NPU 避开物理卡6/7并先核实占用，不停止服务或重置设备。
