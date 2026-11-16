#!/usr/bin/env bash
set -Eeuo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PLATFORM=ascend
export http_proxy=http://127.0.0.1:7898
export https_proxy=http://127.0.0.1:7898
export HTTP_PROXY=http://127.0.0.1:7898
export HTTPS_PROXY=http://127.0.0.1:7898
UCM_HOME=${UCM_HOME:-/home/qyh/dspark-0731-debug-20261116/ucm-26464d0}
python -m pip install -v -e "${UCM_HOME}" --no-build-isolation
