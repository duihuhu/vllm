#!/bin/bash
SERVER_PORT=${1:-8082}
# MODEL_PATH=${2:-/workspace/file/models/llama2-13b/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1/}
MODEL_PATH=${2:-/workspace/file/models/llama2-7b/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/}

python ./vllm/entrypoints/server.py  --model ${MODEL_PATH} \
    --local_host 0.0.0.0 --local_port ${SERVER_PORT} \
    --worker-use-ray  --tensor-parallel-size 1 \
    --block-size 16 --enable-separate \
    --role=prompt --enable-direct \
    # --enable-layer --enable-dcache --enable-radix-caching \