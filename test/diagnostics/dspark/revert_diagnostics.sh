#!/usr/bin/env bash
set -Eeuo pipefail
DIAG_ROOT=${DIAG_ROOT:-/home/qyh/dspark-0731-debug-20261116}
VLLM_ASCEND_HOME=${VLLM_ASCEND_HOME:-/vllm-workspace/vllm-ascend}
UCM_HOME=${UCM_HOME:-${DIAG_ROOT}/ucm-26464d0}
git -C "${VLLM_ASCEND_HOME}" apply --reverse "${DIAG_ROOT}/diagnostics/ascend-nan-diagnostic.patch"
git -C "${UCM_HOME}" apply --reverse "${DIAG_ROOT}/diagnostics/ucm-nan-diagnostic.patch"
echo "diagnostics reverted from ${VLLM_ASCEND_HOME} and ${UCM_HOME}"
