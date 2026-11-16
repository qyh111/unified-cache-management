#!/usr/bin/env bash
set -Eeuo pipefail
DIAG_ROOT=${DIAG_ROOT:-/home/qyh/dspark-0731-debug-20261116}
VLLM_ASCEND_HOME=${VLLM_ASCEND_HOME:-/vllm-workspace/vllm-ascend}
UCM_HOME=${UCM_HOME:-${DIAG_ROOT}/ucm-26464d0}
test -d "${VLLM_ASCEND_HOME}/.git" || { echo "not a git checkout: ${VLLM_ASCEND_HOME}"; exit 1; }
test -f "${UCM_HOME}/ucm/integration/vllm/hma_connector.py" || { echo "UCM source not found: ${UCM_HOME}"; exit 1; }
test -z "$(git -C "${VLLM_ASCEND_HOME}" status --short)" || { echo "checkout is not clean: ${VLLM_ASCEND_HOME}"; git -C "${VLLM_ASCEND_HOME}" status --short; exit 1; }
git -C "${VLLM_ASCEND_HOME}" apply --check "${DIAG_ROOT}/diagnostics/ascend-nan-diagnostic.patch"
git -C "${UCM_HOME}" apply --check "${DIAG_ROOT}/diagnostics/ucm-nan-diagnostic.patch"
git -C "${VLLM_ASCEND_HOME}" apply "${DIAG_ROOT}/diagnostics/ascend-nan-diagnostic.patch"
git -C "${UCM_HOME}" apply "${DIAG_ROOT}/diagnostics/ucm-nan-diagnostic.patch"
echo "diagnostics applied to ${VLLM_ASCEND_HOME} and ${UCM_HOME}"
