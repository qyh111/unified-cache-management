# Hybrid r3：交给验证 agent 的测试指南

## 任务与固定基线

只执行验证并提交报告，不开发 PP/MTP，不修改 store，不自行降低模型层数或篡改原生 KV specs 以制造 PASS。

- 仓库：https://github.com/qyh111/unified-cache-management.git
- 分支：dev_qyh_0928
- **被测源码 commit：323aa0da37985c421b8d276eda81a1910509b362**。
- 本指南可能由后续纯文档提交提供；被测实现仍固定上述 SHA。
- 当前仅离线测试通过：Hybrid 29、原布局33、工具24。旧round1–6是r1证据，不能作为r3的runtime结果。
- 先CPU/simu与NPU/ascend两条路径。CPU不是CUDA硬件验证；CUDA没有设备则NOT_RUN。
- PP/PCP/DCP=1、MTP关闭；不安排Eagle、真实推理或独立store字节功能测试。

## 环境与源码

历史环境为110.138.0.3上的codex_kvcache_ascend_20260905容器，仅作为定位线索，先确认实际可用性。历史CPU为vLLM0.29.0+cpu，NPU为vLLM0.26.0+empty与Ascend0.19.1rc2.dev1373+gcf0baa38d；记录实装版本，不擅自升级引擎。

每条路径用独立新目录、venv/native库和cache；保留所有旧结果，不覆盖共享环境。已有venv可用于核对依赖，但安装/构建在独立验证环境进行。

```bash
COMMIT=323aa0da37985c421b8d276eda81a1910509b362
# RUN_ROOT 用本次唯一新目录；CPU/NPU分别创建，以下示例为CPU。
RUN_ROOT=/home/qyh/ucm_hybrid_r3_cpu_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT"
git clone --branch dev_qyh_0928 --single-branch https://github.com/qyh111/unified-cache-management.git "$RUN_ROOT/src"
cd "$RUN_ROOT/src"
git checkout --detach "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT" || exit 1
git status --porcelain > "$RUN_ROOT/git-status-before.txt"
```

在有相应引擎依赖的独立venv内构建：CPU设`PLATFORM=simu`，NPU按硬件设`PLATFORM=ascend`或`ascend-a3`，先加载CANN环境。然后：

```bash
python -m pip install -v -e . --no-build-isolation --no-deps
python -m pip install -e ./toolkit --no-deps
export PYTHONPATH="$RUN_ROOT/src:$RUN_ROOT/src/toolkit${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import ucm; from ucm_toolkit.tools.model_check import common; print(ucm.__file__); print(common.__file__)' | tee "$RUN_ROOT/imports.log"
python -m pip freeze > "$RUN_ROOT/packages.txt"
python test/suites/Unit/test_ucm_hybrid.py
python test/suites/Unit/test_kv_cache_layout.py
python -m unittest discover -s toolkit/tests -p 'test_model_check*.py'
```

缺少构建依赖时按现有项目环境准备并记录；不要替换引擎。若复用旧native二进制，记录文件SHA、来源commit和构建平台；报告明确“Python来自本提交，native复用”，不能声称完整重建。确认实际加载库没有串CPU/NPU。使用`ucm-toolkit`，不要用不存在的`python -m ucm_toolkit`入口。

## 用例函数

在对应venv激活且RUN_ROOT设定后定义。NPU另先执行`source /usr/local/Ascend/ascend-toolkit/set_env.sh`；设备号必须选当前空闲卡，不能照抄历史4/5或停止他人任务。串行运行。

```bash
run_case() {
  local case_name="$1" model="$2" tokens="$3" tp="$4" capacity="$5" mode="$6"
  shift 6
  local case_dir="$RUN_ROOT/results/$case_name"
  if [ -e "$case_dir" ]; then echo "Use a new case name: $case_dir"; return 2; fi
  mkdir -p "$case_dir/layout" "$RUN_ROOT/cache/$case_name"
  local cmd=(ucm-toolkit run model-check --hybrid --model "$model"
    --tokens "$tokens" --block-size 128 --tp "$tp"
    --store-pipeline 'Cache|Posix' --storage-backends "$RUN_ROOT/cache/$case_name"
    --cache-buffer-capacity-gb "$capacity" --cache-load-exclusive-buffer-number 512
    "$mode" "$@")
  printf '%q ' "${cmd[@]}" > "$case_dir/command.sh"
  printf '\n' >> "$case_dir/command.sh"
  UCM_HYBRID_DUMP_LAYOUT="$case_dir/layout" "${cmd[@]}" > "$case_dir/run.log" 2>&1
  local rc=$?
  printf '%s\n' "$rc" > "$case_dir/exit-code.txt"
  echo "$case_name exit=$rc log=$case_dir/run.log"
  return "$rc"
}
```

不要用全局`set -e`导致失败后跳过证据收集。每个case结束检查日志，严重初始化失败先定位再继续，不循环盲试。

## P0：先验证分流与group key

模型材料历史路径（先确认存在）：

```bash
QWEN=/home/models/Qwen3.8-27B-w8a8
GLM=/home/qyh/kv-ascend/models/glm52
MM3=/home/qyh/mm_m3_material
```

这些路径的名字不代表本次实际启用了权重量化。记录config及必要tokenizer/自定义配置代码SHA，不下载完整权重。

CPU执行：

```bash
run_case qwen-cpu-tp2-layerwise "$QWEN" 6144 2 16 --layerwise
run_case qwen-cpu-tp2-bulk "$QWEN" 6144 2 80 --no-layerwise
```

NPU执行（先设置DEVICES为两张确认空闲的卡，例如历史曾用4,5）：

```bash
run_case qwen-npu-tp2-layerwise "$QWEN" 6144 2 16 --layerwise --device-id "$DEVICES"
run_case qwen-npu-tp2-bulk "$QWEN" 6144 2 80 --no-layerwise --device-id "$DEVICES"
```

16/80是起始配置，80来自历史bulk经验，不表示r3必需80。容量可按日志所需调节，一般每私有buffer不超过128GiB；先核算TP总量和当前可用主机内存。不要改store实现或偷偷缩减原生模型。资源不足记BLOCKED_CAPACITY。

Qwen验收点：

- 每rank都必须有独立layout文件，rank集合完整；模型实际有4个原生group，每group16层（64层配置），不要硬编码FA是group0：CPU历史FA为group3。
- CPU为`native_state_page`，无conv/data0/data1 ghost槽；`padding.groups[*].padding_bytes=0`表示connector新增空槽为零，完整page仍包含引擎已有尾部。
- CPU历史TP2 page=1,835,008 B、State有效内容=1,603,584 B，只是同引擎同配置的参考。若不同，记录实际spec/view/stride，不改数据凑数字。
- NPU为`state`，layerwise参考槽位[30720,1572864,1572864]；FA第一槽空，State最后槽空。每group16行，不能出现旧版FA16+32ghost/State48行。
- bulk每group1行，由本group16层模板组合；四group各有key/plan，不再合并State。namespace为hybrid-v1-r3。
- 保存/恢复必须经过生产Hybrid hooks与真实Cache|Posix；每rank `compared_payload_segments`>0，退出0，无submit/wait错误。比较数不能沿用旧版硬编码。

## P1：GLM与MiniMax回归

P0通过后：CPU/NPU各跑GLM TP2与MiniMax TP1的layerwise/bulk。NPU选卡数量与TP一致；CPU不传device-id。

```bash
run_case glm-cpu-tp2-layerwise "$GLM" 1024 2 16 --layerwise
run_case glm-cpu-tp2-bulk "$GLM" 1024 2 16 --no-layerwise
run_case mm3-cpu-tp1-layerwise "$MM3" 1024 1 16 --layerwise
run_case mm3-cpu-tp1-bulk "$MM3" 1024 1 24 --no-layerwise
# NPU同样参数，case名换成npu，并增加 --device-id "$DEVICES" 或单张卡 "$DEVICE"。
```

GLM只有运行时实际存在indexer才会选择shared_indexer；仅普通配置所有层有BF16 Indexer，不算Shared/LI C8验收。MiniMax layerwise应选择minimax_m3，dense层index空；单group bulk的compact无层间ghost是预期。

## P2：GLM真实Shared与C8覆盖

材料和引擎支持时再跑，禁止用改小模型或手工删tensor制造覆盖。查看真实`indexer_types`、量化材料和注册views；additional_config有开关不等于实际生效。

```bash
# 示例：双C8开关；材料路径指向实际相应配置。TP2与两张空闲NPU。
run_case glm-npu-c8-tp2-layerwise "$GLM_C8" 1024 2 16 --layerwise   --device-id "$DEVICES"   --additional-config '{"enable_sparse_sfa_c8":true,"enable_sparse_li_c8":true}'
```

分别覆盖SFA关/开，LI关/开，以及实际full-C8、mixed BF16/C8、Shared层。A2/A3/A4、128tokens参考：

| 场景 | layerwise固定槽位B |
|---|---|
| SFA关、LI关 | [131072,16384,32768] |
| SFA开、LI关 | [83968,32768] |
| SFA关、全LI C8 | [131072,16384,16384,256] |
| SFA开、全LI C8 | [83968,16384,256] |
| SFA关、mixed | [131072,16384,16384,16384,256] |
| SFA开、mixed | [83968,16384,16384,256] |

mixed的C8层index_tail为空、BF16层scale为空、Shared层index/index_tail/scale为空。全C8没有index_tail。scale按实际dtype确定，A5可为512B，不硬写256。bulk单group紧凑排列真实segments。

目前工具没有`--quantization`参数，不能直接拼serve参数；所需量化构造不支持则记BLOCKED_TOOL/NOT_COVERED，留下证据交回修复。没有GLM_C8材料则记BLOCKED_MATERIAL，不把普通GLM PASS冒充C8。

## 报告与失败处理

回传独立结果目录，例如`results_remote_20260929_r3/`，不覆盖之前round目录。

每个case保存command.sh、exit-code.txt、完整run.log、每rank layout、模型config身份、原生KVCacheConfig/view/stride（若当前日志没有则说明缺少，另作只读诊断采集）。全局保存完整commit、前后git status、依赖版本、导入路径、native来源、设备映射、主机内存快照。

REPORT.md按表列：路径/模型材料/TP/模式/policy/group数/每group行数/实际槽位/有效数据与padding及alignment/容量/各rank比较数/状态/证据路径。

状态区分PASS、FAIL、BLOCKED_ENV、BLOCKED_CAPACITY、BLOCKED_TOOL、BLOCKED_MATERIAL、NOT_RUN。PASS仅声明无权重布局与dump/load恢复，不声明推理精度、量化计算正确性、MTP或CUDA硬件覆盖。

若native page校验拒绝，保留完整异常和view/allocation/descriptor，不能移除边界检查或回退Ascend槽位。确需诊断补丁时单独保存diff，标dirty调试结果；正式验收需补丁合回并推送新commit后重跑。禁止混入旧源码路径或手工覆盖当前基线。
