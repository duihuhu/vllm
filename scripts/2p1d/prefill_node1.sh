#!/bin/bash
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eno2
SERVER_HOST=${1:-10.156.154.242}
SERVER_PORT=${2:-8082}
# MODEL_PATH=${3:-/workspace/file/models/llama2-13b/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1/}
MODEL_PATH=${3:-/workspace/file/models/llama2-7b/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/}
GS_HOST=${4:-10.156.154.242}
GS_PORT=${5:-8081}

python ./vllm/entrypoints/server.py --model ${MODEL_PATH} \
    --local-host ${SERVER_HOST} --local-port ${SERVER_PORT} --cluster-rank 0 \
    --gs-host ${GS_HOST} --gs-port ${GS_PORT} \
    --worker-use-ray  --tensor-parallel-size 1 \
    --block-size 16 --enable-separate \
    --role=prompt --enable-direct \
    --ray-address 10.156.154.242:6379 \
    2>&1 > logs/prefill1.log
    # --enable-layer --enable-dcache --enable-radix-caching \