#!/usr/bin/env bash
set -euo pipefail

MODEL="/mnt/nvme/fyf/models/DeepSeek-V2-Lite"
LOG_FILE="/mnt/nvme/fyf/proj2/log/start_log.txt"

RAY_HOST="127.0.0.1"
RAY_PORT="9305"

export RAY_ADDRESS="${RAY_HOST}:${RAY_PORT}"
export VLLM_SERVER_DEV_MODE=1
export RAY_DEDUP_LOGS=0
export VLLM_EEP_PROFILE_CSV=/mnt/nvme/fyf/proj2/log/424/eep_profile.csv
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$LOG_FILE")"

# 如果本机没有固定 Ray 实例，就起一个。
# 如果你机器上可能有别的 Ray 任务，不建议自动 ray stop。
if ! ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; then
    ray start --head --port="$RAY_PORT" --disable-usage-stats
fi

vllm serve "$MODEL" --trust-remote-code --enable-sleep-mode \
    --host 0.0.0.0 \
    --port 8005 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --enable-eplb \
    --enable-elastic-ep \
    --all2all-backend nixl_ep \
    --eplb-config.num_redundant_experts 64 \
    --data-parallel-backend ray \
    --distributed-executor-backend ray \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --data-parallel-rpc-port 9876 \
    --data-parallel-start-rank 0 \
    --enforce-eager \
    2>&1 | stdbuf -oL -eL sed -u \
        -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' \
        -e 's/(APIServer pid=[0-9]*) *//g' \
        -e 's/(DPMoEEngineCoreActor pid=[0-9]*) *//g' \
        -e 's/(RayWorkerWrapper pid=[0-9]*) *//g' \
        -e 's/^ *//' \
        | tee "$LOG_FILE"

# stdbuf -oL -eL vllm serve "$MODEL" --trust-remote-code \
#     --host 0.0.0.0 \
#     --port 8005 \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.5 \
#     --max-model-len 4096 \
#     --enable-sleep-mode \
#     --no-enable-prefix-caching \
#     --enable-expert-parallel \
#     --enable-eplb \
#     --enable-elastic-ep \
#     --eplb-config.num_redundant_experts 64 \
#     --data-parallel-size 2 \
#     --data-parallel-size-local 2 \
#     --data-parallel-rpc-port 9876 \
#     --data-parallel-start-rank 0 \
#     --enforce-eager \
#     2>&1 | stdbuf -oL -eL sed -u \
#         -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' \
#         -e 's/(APIServer pid=[0-9]*) *//g' \
#         -e 's/(DPMoEEngineCoreActor pid=[0-9]*) *//g' \
#         -e 's/(RayWorkerWrapper pid=[0-9]*) *//g' \
#         -e 's/^ *//' \
#         | tee "$LOG_FILE"