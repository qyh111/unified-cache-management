# GLM5.3 CPU/NPU 多TP布局与存取验收

日期：2026-09-29。**CPU TP1/2/4、NPU TP1/2 的 layerwise/bulk 已有通过证据；NPU TP4 尚未验收，不能宣布 GLM5.3 全部验完。**

本轮代码：`0699ccf333b9a8fefa1ba36b7b543221cb948b8c`，qyh/dev_qyh_0928。仅新增显式、作用域受限的CPU FP8计算factory mock；connector沿用03b8134d版本。mock一旦执行计算就抛错，不删改模型量化配置、KV spec、dtype、分组或allocation。

## 最终矩阵

每格PASS均要求所有rank输出PASS且进程退出0；分段数按每rank计。

| 路径 | TP | layerwise | bulk | 分段数/rank | 实际逻辑block tokens |
|---|---:|---|---|---:|---:|
| CPU模拟CUDA布局 | 1 | PASS | PASS | 78 | 4352 |
| CPU模拟CUDA布局 | 2 | PASS | PASS | 144 | 2176 |
| CPU模拟CUDA布局 | 4 | PASS | PASS | 254 | 1152 |
| NPU | 1 | PASS（上一轮） | PASS（上一轮） | 156 | 4352 |
| NPU | 2 | PASS | PASS（卡1/5对照） | 288 | 2176 |
| NPU | 4 | 中断，未验收 | 中断，未验收 | — | 未完成布局捕获 |

NPU TP1证据来自 `03b8134d5711682fb44575eeb87f70d9ccb694c6`，见上一份 `results_remote_20260929_glm53/REPORT.md`。本轮14次尝试中8次完整通过，另6次中断；合并上一轮TP1，共10个配置格有通过证据，2个NPU TP4配置格未通过。

所有通过项均运行真实native runner分配、真实scheduler、真实UCMHybridConnector和Cache|Posix，完成源填充、dump、目标外部lookup、目标poison、load与逐字节比对。CPU在已有CUDA布局模拟harness之外显式启用 `UCM_MODEL_CHECK_GLM53_FP8_MOCK=1`；它不是GPU硬件或forward验收。NPU不启用此mock。本轮无完整权重、无forward、无独立store功能测试。

## 环境、参数和证据

- 宿主机 `110.138.0.3`；容器 `ucm-nightly-main-20260929`。
- vLLM源码 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；Ascend源码 `b64b4d714484feaa6ca71edc99b98318fe6d2f0d`。
- tokens13056、配置block128、TP1/2/4，实际block由引擎生成，见上表。
- layerwise buffer16GB；CPU bulk64GB；NPU TP2 bulk通过对照为32GB；exclusive16。容量为每私有buffer，需乘TP核算总内存。
- GLM5.3 config SHA256：`bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`。
- tokenizer_config SHA256：`98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc`。
- 远端全部命令/日志/退出码：`/home/qyh/ucm_hybrid_r5_20260929/glm53-matrix-0699ccf3/`。
- 本地完整证据：工作区 `connector_v2_v1store_0928/results_remote_20260929_glm53_matrix/evidence/`；同目录上一级有复现脚本 `run_matrix.sh` 与 `retry_npu.sh`。仓内摘要见 `docs/connector-glm53-matrix-summary.json`，完整日志另保存在上述远端目录。缓存数据未打包。
- 本地工具单测31项通过；Hybrid32项回归通过。原有布局33项与GLM5.3地址oracle5项上一轮已通过，此轮未修改对应connector实现。

## SIGTERM中断记录：不得忽略

1. NPU TP2 bulk在卡5/7上首次两个rank都已PASS（各288分段），随后清理阶段某rank被SIGTERM终止，整体exit1。此尝试不计完整PASS。
2. 同卡5/7、buffer32GB复测，dump后又被SIGTERM终止，未输出PASS。
3. 卡1/5、同代码、buffer32GB、同断言对照成功：两个rank均PASS且exit0；作为TP2 bulk通过证据。
4. TP4卡4/5/6/7，layerwise/bulk两项都在初始化阶段由local rank2（卡6）先收到SIGTERM；换卡1/4/5/7后两项仍由local rank3（卡7）先收到SIGTERM。均没有完成connector存取，不算PASS。
5. 只读资源检查时宿主机可用内存约1.8TiB，无本次测试残留活进程；未见明确的connector Python异常。卡映射相关性不构成硬件故障结论，信号来源仍未知；未重置设备、未停止其他服务、未更改信号处理掩盖问题。

下一位agent优先定位首个SIGTERM来源，区分原生runtime/驱动、外部进程管理和退出清理问题；torchrun向其他rank发出的SIGTERM是后续动作，不是根因。确认四张可用卡后复测TP4两种模式。仓内验收指南含命令和标准。

## 验收边界与交接

目前只完成表中指定布局/存取范围；真实GPU硬件、TP8、PP/PCP/DCP、MTP、真实forward/tail数值重建、State跨轮复用与0.30 checkpoint/CoW仍未验收。State完整block保存继续沿用wait_for_save，不启用任意checkpoint导出。

后续工作见仓内 `docs/connector-glm53-and-030-acceptance.md`：先解决GLM5.3 NPU TP4中断，再推进DSV4-Flash/Qwen3.8-Flash-Next的0.30布局model-check，并回归GLM5.2/MiniMax-M3/Qwen3.8。store实现与完成发布语义不变，必要变更先与用户对齐。
