#!/bin/bash
SERVER_PORT=${1:-8081}
# MODEL_PATH=${2:-/workspace/file/models/llama2-13b/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1/}
MODEL_PATH=${2:-/workspace/file/models/opt-125m/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/}

python3 ./vllm/global_scheduler/gs/async_global_scheduler.py --model ${MODEL_PATH} \
    --host 0.0.0.0 --port ${SERVER_PORT} \
    --ep-policy random --ed-policy random --enable-separate \