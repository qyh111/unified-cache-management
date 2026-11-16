#!/usr/bin/env bash
set -Eeuo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
DIAG_ROOT=${DIAG_ROOT:-/home/qyh/dspark-0731-debug-20261116}
UCM_HOME=${UCM_HOME:-${DIAG_ROOT}/ucm-26464d0}
export PYTHONPATH=${UCM_HOME}:${PYTHONPATH:-}
export ENABLE_UCM_PATCH=1
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_COMPUTE_NANS_IN_LOGITS=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_DSPARK_DIAG=${VLLM_ASCEND_DSPARK_DIAG:-0}
export VLLM_ASCEND_DSPARK_DIAG_EVERY=${VLLM_ASCEND_DSPARK_DIAG_EVERY:-1}
export VLLM_ASCEND_DSPARK_DIAG_MAX_LAYERS=${VLLM_ASCEND_DSPARK_DIAG_MAX_LAYERS:-0}
export VLLM_ASCEND_DSPARK_DIAG_MAX_BLOCKS=${VLLM_ASCEND_DSPARK_DIAG_MAX_BLOCKS:-1}
export UCM_LOG_PATH=${DIAG_ROOT}/logs/ucm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL=${MODEL:-/data/model/DeepSeek-V4-Flash-0731-w8a8}
PORT=${PORT:-18080}
GPU_MEM=${GPU_MEM:-0.70}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
UCM_CONFIG_FILE=${1:-${DIAG_ROOT}/diagnostics/ucm_posix.yaml}
LOG_FILE=${LOG_FILE:-${DIAG_ROOT}/logs/ucm-diagnostic.log}
mkdir -p "$(dirname "${LOG_FILE}")" "${UCM_LOG_PATH}"
exec > >(tee -a "${LOG_FILE}") 2>&1
KV_TRANSFER_CONFIG=$(cat <<JSON
{
  "kv_connector": "UCMConnector",
  "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "UCM_CONFIG_FILE": "${UCM_CONFIG_FILE}"
  }
}
JSON
)

exec vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens 8192 \
    --served-model-name dsv4 \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --block-size 128 \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config "${KV_TRANSFER_CONFIG}"
