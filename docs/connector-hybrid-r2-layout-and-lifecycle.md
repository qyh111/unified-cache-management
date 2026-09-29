# Hybrid r2 布局与生命周期清单

2026-09-29。r2 已完成首版代码和本地测试，目标环境 Model-check 尚未运行。旧六轮仅覆盖 r1。

## 布局与 key

- [x] 每个原生 group 独立路由、key、block table 和存储记录。
- [x] 16 字节 key：前14字节 prefix hash + 2字节 tag；tag 为 type(2)、group(4)、tp(4)、pp(4)、reserved(2)。group_id 限0–15，超出拒绝；实际 TP 隔离沿用旧 RequestHasher/rank consistency。
- [x] namespace 升级 hybrid-v1-r2；纳入有序 group/layer/spec 身份，排除设备地址和 allocation 容量。r1 缓存不可复用。
- [x] FA 多 group 取共同前缀；State 多 group 求共同边界，不能仅取各组最近命中的最小值。
- [x] State 策略：conv、data0、data1；FA 为 ghost/K/V，State 为 conv/state/ghost。packed FA 拆为两个字节范围，不声称每范围一定是 K/V；combined State 按原生 shapes/dtypes 拆分。
- [x] Shared Indexer 策略：attention/index/scale；mixed BF16/LI C8 增 index_tail，拆分保留 stride；Shared 层补对应空槽。全 C8 不预留 BF16 尾段。
- [x] MiniMax-M3 识别 index_cache 角色，dense 层 Indexer 槽为空；普通 attention 要求组件长度一致。
- [x] 各 group 本地层数必须相同，不等层数明确拒绝，不隐式补大量空行。
- [x] Model-check dispatch 比较包含 group_id；诊断包含 policy、group 类型与各组 rows。
- [ ] 从新远端 commit 验证 Qwen 四 group、CPU combined State 与 NPU tuple State；TP1/2、bulk/layerwise。
- [ ] GLM SFA/LI 开关、mixed/full-C8/Shared 与 MiniMax dense/sparse runtime 覆盖，记录槽位、各组 keys、空间、恢复字节比较。

Qwen TP2：16层/group，槽位[30720,1572864,1572864]，每 group block 为50,823,168 B，约48.47 MiB；layerwise为16 shard。必须按真实FA prefix/State boundary访问量统计总空间，不能只比较单key大小。

## P1：block_pool 与保存源生命周期

本地参考：vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/。
connector.py 的 bind_gpu_block_pool 转发到 scheduler；scheduler.py 的 _pinned_saves 按 store_job_id 记录 block IDs 和未完成 worker 数。发任务前 pool.touch，update_connector_output 收齐 completed_saves 后 pool.free_blocks。
worker.py 的 wait_for_save 记录 CUDA event 并入队，没有在此等待全部传输。发送线程 finally 调用 finish_store_job；任务结束不等于保存成功或 durable commit。

- [ ] 对照目标 CPU vLLM/Ascend 的实际 SPI，不直接照搬本地较新 Mooncake。
- [ ] 区分回收再分配与同一请求下一步原地更新；引用计数只保护前者，不自动冻结 State/SWA 内容。
- [ ] 核实 FA、State align snapshot、SWA 窗口的可变性、null block、共享 block 和源读取结束时刻。
- [ ] 在确认可安全保留源 block 前，Hybrid wait_for_save 保持排空全部 dump；FAWA/SWA 当前路径另审计。
- [ ] 若实现异步保存：scheduler 发任务前持有精确引用，worker 回报源读取结束，收齐参与 rank 后只释放一次；覆盖失败、取消、抢占、请求结束和引擎空闲仍有任务。
- [ ] 测试立即复用、原地覆盖、重复/迟到完成、失败释放、退出清理，防止引用泄漏和提前释放。

## P1：worker metadata 契约

现有 UCMWorkerMetadata：is_mla、load_failed_reqs、missing_reqs、missing_blocks、dump_succeeded_blocks。失败/缺失取并集；Hybrid 按非MLA路径取dump成功交集。build_connector_worker_meta 输出累计值后重置；当前没有 block_pool store_job_id 完成协议。

- [ ] 明确 scheduler→worker 的任务身份、request/group、block IDs、参与 rank；worker→scheduler 区分load失败/缺失、dump成功、源block可释放。
- [ ] 不用 dump_succeeded_blocks 代替任务完成；失败任务也需释放资源，源读取完成不提升为 durable commit。
- [ ] 独立group keys的成功/缺失报告覆盖所需group；PP/TP聚合明确参与集合，不将缺rank算作成功。
- [ ] 若新增完成字段，使用job_id与rank身份去重，覆盖部分提交失败及跨step完成。
- [ ] 核实 build_connector_worker_meta/aggregate/update_connector_output 调用时序，测试重复、迟到、取消后回报。

## 后续顺序

先验证r2布局，再审计生命周期/metadata并补测试；PP/MTP、DCP、PCP仍在剩余计划中。store/native与旧发布语义不改，涉及修改先对齐。这些任务可由后续agent继续，不要求用户亲自实现。
