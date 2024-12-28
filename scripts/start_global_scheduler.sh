#!/bin/bash
SERVER_HOST=${1:-10.156.154.242}
SERVER_PORT=${2:-8081}
# MODEL_PATH=${2:-/workspace/file/models/llama2-13b/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1/}
MODEL_PATH=${3:-/workspace/file/models/llama2-7b/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/}

python3 ./vllm/global_scheduler/gs/async_global_scheduler.py --model ${MODEL_PATH} \
    --host ${SERVER_HOST} --port ${SERVER_PORT} \
    --ep-policy random --ed-policy random --enable-separate \