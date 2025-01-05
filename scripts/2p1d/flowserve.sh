#!/bin/bash 

MODEL_PATH=${1:-/workspace/file/models/llama2-7b/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/}
DATASET_PATH=${2:-/workspace/file/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}
DATASET=${3:-ShareGPT}
GS_HOST=${4:-10.156.154.242}
GS_PORT=${5:-8081}

python3 ./benchmarks_flowserve/main.py \
    --tokenizer-path ${MODEL_PATH} \
    --dataset-path ${DATASET_PATH} \
    --gs-host ${GS_HOST} --gs-port ${GS_PORT} \
    --test-type open --dataset ${DATASET} --request-rate 1 --num-requests 16