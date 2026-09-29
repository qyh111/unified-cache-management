# GLM5.3 与后续 0.30 模型验收交接

最新状态见 [GLM5.3矩阵报告](connector-glm53-model-check-results.md)：CPU TP1/2/4、NPU TP1/2的两种模式已有通过证据；NPU TP4因SIGTERM仍未验收，优先排查。

## 范围与约束

代码仓库 `https://github.com/qyh111/unified-cache-management.git`，分支 `dev_qyh_0928`。开始时读取当前最新报告，固定完整 SHA；不要仅凭分支名判断测试版本。保留未提交文件，禁止在 dev_connector 开发。

当前 GLM5.3 接入基线 `0699ccf3`，矩阵结果见工作区 `connector_v2_v1store_0928/results_remote_20260929_glm53_matrix/`。较早 NPU TP1 layerwise/bulk 在 `03b8134d5711682fb44575eeb87f70d9ccb694c6` 通过，各比对156个有效载荷分段。0699ccf3 仅在工具中新增显式 CPU FP8 计算 mock，不改 connector。

只验证布局/block存取时允许 mock 计算 kernel 选择；不能 mock spec、分组、压缩比、dtype、allocation/reshape、manager、block ID 或 UCM lookup。误执行计算 mock 必须报错。CPU 是模拟 CUDA 布局路径，不是 GPU 硬件验收。没有模型权重时不能宣称推理精度通过。

Pipeline 固定 Cache|Posix，一个 Hybrid store；FA/SWA 继续 FAWA 两 store。不要修改 store 接口或完成发布语义，必要修改先和用户对齐。无需另做独立真实 store 功能测试。

## 环境与执行方式

宿主机 `root@110.138.0.3`，容器 `ucm-nightly-main-20260929`。

- 测试根目录 `/home/qyh/ucm_hybrid_r5_20260929`。
- CPU：`cpu/src` + `cpu/venv/bin/ucm-toolkit`，UCM native simu。
- NPU：`npu/src` + `npu/venv/bin/ucm-toolkit`，UCM native ascend。
- vLLM source `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；Ascend source `b64b4d714484feaa6ca71edc99b98318fe6d2f0d`。记录实际 version 字符串及源码 SHA，不把 main build 当 release。
- NPU 先 source `/usr/local/Ascend/ascend-toolkit/set_env.sh`，不要随后覆盖 CANN 的 PYTHONPATH。
- 运行前检查 npu-smi，只用空闲设备，不停止其他人的服务；各 case 顺序运行，避免端口和设备竞争。

参考命令（容器内执行；替换 case 路径保证每项独立）：

```bash
root=/home/qyh/ucm_hybrid_r5_20260929
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$root/npu/src"
git rev-parse HEAD
"$root/npu/venv/bin/ucm-toolkit" run model-check \
  --hybrid --model "$root/materials/glm53" \
  --tokens 13056 --block-size 128 --tp 2 --device-id 5,7 \
  --store-pipeline 'Cache|Posix' --storage-backends /path/to/unique-case/cache \
  --cache-buffer-capacity-gb 16 --cache-load-exclusive-buffer-number 16 \
  --layerwise
```

bulk 用 `--no-layerwise`，GLM5.3 TP1 需要 buffer64GB：原生 store 对约54MiB shard要求至少55GB。默认容量16GB，按需增加，单私有buffer一般不超过128GB；总内存须乘TP rank数。CPU使用cpu目录/venv，并显式 `export UCM_MODEL_CHECK_GLM53_FP8_MOCK=1`。此开关仅允许GLM5.3模型；其他模型不要开启。

## 每项通过标准

1. 保存完整 SHA、环境版本、模型材料来源及哈希、完整命令、stdout/stderr、退出码；TP每个rank都有PASS，不能只看rank0或exit0。
2. 源请求由真实 scheduler 分配；dump key非空，目标UCM外部lookup非空，目标实际load分段逐字节比对通过。工具应保持源/目标block不重叠和目标poison检查，不得关闭断言。
3. 0.30目标测试请求使用原生skip_reading_prefix_cache，防止HBM命中绕过UCM；不能把这个测试选项当产品行为变更。
4. 检查CUDA/NPU原生group数、每组层数、token block、descriptor与slot/padding报告。tail保留原生group但不在持久化路由，State不合并，namespace的scheduler/worker口径一致。
5. 独立地址oracle和model-check互补：后者用connector布局产生存取范围，不能独自证明物理解析正确。执行仓内GLM5.3捕获布局测试。

```bash
python test/suites/Unit/test_ucm_hybrid.py
python test/suites/Unit/test_glm53_layout.py
python test/suites/Unit/test_kv_cache_layout.py
python -m unittest discover -s toolkit/tests -p 'test_model_check*.py'
```

## GLM5.3后续生命周期/推理验收

多rank中断必须先看最新矩阵报告。换卡仅是对照，不构成硬件故障诊断；SIGTERM原因未查清时不能宣称模型或环境已验收。

model-check合成数据通过后仍需单独验证以下内容，缺环境时写BLOCKED和原因：

- 跨轮State原地更新：人为延迟异步dump，确认wait_for_save返回前源State不被下一轮写入；引用block_pool不等价于防写。
- 不完整命中：在独立测试命名空间模拟某个State组或某个TP rank缺失，确认共同恢复边界回退；FA单独命中不能代替State快照。不得改生产缓存或放宽一致性判断。
- preemption、请求结束和并发复用：已完成/失败load与save的worker metadata、task/event清理、block释放时机；出现失败不能发布可命中但数据不完整的记录。
- 0.30 checkpoint/CoW：核对真实boundary_state_offloads及copy完成顺序。当前只允许完整block导出，任意checkpoint导出未实现，不能自行解除限制。
- tail数值：恢复长度T为4的倍数，比较冷启动与恢复后首个pool、跨launch分块decode/prefill的indexer结果。T非4倍数作为拒绝/不命中对照，不能丢tail后仍声称可恢复。
- 若有完整权重，再比较冷启动与prefix恢复后的logits/输出，覆盖短/长prompt、并发、重复命中。禁用全部计算mock，记录量化模式和误差阈值依据。
- PP/PCP/DCP>1、MTP仍未验收且有guard；不要把TP结果推广过去。Eagle不在本轮范围。

## 后续0.30模型：按问题分开推进

### DeepSeek V4 Flash

材料 `/home/qyh/ucm_hybrid_r5_20260929/materials/dsv4` 来自 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 的config，缺量化描述时不能标W8A8完整验收。

已知CPU FAWA旧解析把4D tensor错误按axis1检查token，真实形状如 `(1590,1,64,584)`；NPU缺 `_C_ascend.npu_sparse_attn_sharedkv` 计算扩展。先核对真实backend约束/压缩子页/SWA tail，仅mock不用执行的计算选择。保留FAWA两store，不套GLM5.3策略。CPU首轮参考block256、kv-cache-dtype fp8，CLI不传`--hybrid`；以实际配置路由为准。

### Qwen3.8-Flash-Next

材料 `/home/qyh/ucm_hybrid_r5_20260929/materials/qwen38-next`，官方材料 revision 以 r5 报告和资产记录的完整 SHA 为准。已知QSA模型构造要求FlashAttention，NPU当前路径引用NVIDIA实现。先确认0.30支持分支及真实spec/allocation，不能为了过构造直接替换成旧Qwen3.8布局。

### 旧模型回归

保持GLM5.2 Shared/SFA-C8/LI-C8组合、MiniMax-M3、Qwen3.8/Mamba在各自已验证引擎上的布局与存取。按已有r3/r4/r5报告复用真实配置，覆盖TP1/2、layerwise/bulk。真实Qwen3.8量化权重路径 `/home/models/Qwen3.8-27B-w8a8` 需先确认当前容器可见；它不能代替Flash-Next或GLM5.3推理验收。

交付报告分开列 PASS / FAIL / BLOCKED / 未运行，并附逐rank证据。发现实现问题可以在dev_qyh_0928修复并单测、提交推送，然后按新SHA复测；不得热改远端源码后只报分支名。


## 可直接交给后续 agent 的任务

先读本指南、最新GLM5.3矩阵报告，以及工作区PROJECT_CONTEXT/README/HANDOFF；以最新报告和用户约束覆盖历史过时结论。检查Git与环境，保留未提交内容，固定远端代码SHA。

优先处理最新矩阵报告中NPU多rank的SIGTERM中断：保留日志，定位首个收到信号的rank及信号来源，不把torchrun关闭其他rank的SIGTERM当根因，不因已出现PASS而忽略非零退出码。不要修改信号处理以掩盖中断。然后复核GLM5.3剩余生命周期项目的可执行性，不能把缺权重/缺算子记成PASS。然后按上面的已知阻塞推进DSV4-Flash和Qwen3.8-Flash-Next的0.30布局model-check。只mock未执行的计算，不改变真实布局来源；保持旧GLM5.2/MiniMax-M3/Qwen3.8回归。需要改store时先与用户对齐，其他connector/tool修复在dev_qyh_0928完成测试、提交、推送后再远端验证。输出逐case逐rank报告、命令、日志、SHA和明确的未验收范围。
