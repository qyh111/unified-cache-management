# r4 namespace修复与重测

## r3结果

被测323aa0da：Qwen TP2 CPU/NPU×bulk/layerwise 4/4 PASS；GLM TP2、MiniMax TP1 CPU/NPU×两模式8/8 FAIL。GLM Shared/C8未运行。

原始报告位于工作区connector_v2_v1store_0928/results_remote_20260929_r3/REPORT.md；未修改历史证据。抽查CPU GLM/MM3正式日志均存在两个r3 namespace，Qwen NPU只有一个且比较192/rank。

根因是namespace identity包含逐层str(kv_cache_spec)，而原生generate_scheduler_kv_cache_config将UniformTypeKVCacheSpecs替换为单一代表spec。worker仍持有完整异构spec，导致两个角色计算出不同存储目录。group key本身相同不能抵消目录不同。这是connector身份构造问题，不应在store或harness中绕过。

## 修复

storage_namespace集中构造双方共享的身份：模型配置、device、model/cache dtype、TP、包版本、additional_config、quantization，以及group id/token粒度/State类型/层名集合。排除逐层物理spec、capacity和地址。

前缀改为hybrid-v1-r4；group key和padding算法不变。旧r3目录不能迁入新namespace复用。

本地Hybrid30项、工具24项通过。新增回归模拟worker的异构层spec与scheduler代表spec，并验证两端namespace一致；C8开关、quantization、dtype、TP、模式和group身份变化仍隔离。此为离线回归，尚未做修复后的runtime验收。

## 交给验证agent

以包含本文件的修复commit为源码基线（实施方提供完整SHA），从qyh/dev_qyh_0928新检出；记录完整SHA和干净状态。沿用r3测试指南中的环境、模型和命令参数，使用新RUN_ROOT、新cache、新结果目录results_remote_20260929_r4。不是继续测试323aa0da。

1. 优先重跑GLM TP2、MiniMax TP1，CPU/NPU各layerwise/bulk，共8项。
2. 每项确认scheduler与worker使用相同r4 namespace，lookup产生非空load计划，每rank真实恢复比较>0且退出0；dump正常本身不算PASS。
3. Qwen TP2两路径两模式4项防回归，CPU仍native_state_page，NPU仍State语义槽位，四group独立存储。
4. 保持串行，避免torchrun默认端口冲突；不要CPU/NPU并发复用29500。保留失败用例，不覆盖原r3结果。
5. 全部基础项通过后再讨论GLM_C8材料与SFA/LI/Shared覆盖；目前继续NOT_RUN，不将普通BF16结果外推。

store不修改；no-weight Model-check范围不扩大为推理精度验收。若仍分裂namespace，先采集两侧输入身份差异并回传，不移除模型/量化隔离字段来强行凑同一目录。
