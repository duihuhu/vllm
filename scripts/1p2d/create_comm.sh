#!/bin/bash

PREFILL_1_HOST=10.156.154.242
PREFILL_1_PORT=8082
PREFILL_1_RANK="0"




DECODE_1_HOST=10.156.154.20
DECODE_1_PORT=8083
DECODE_1_RANK="1"

DECODE_2_HOST=10.156.154.20
DECODE_2_PORT=8085
DECODE_2_RANK="2"

python  ./vllm/global_scheduler/client/create_comm_test.py \
    --prefill-host ${PREFILL_1_HOST} --prefill-port ${PREFILL_1_PORT} --prefill-rank ${PREFILL_1_RANK} \
    --decode-host ${DECODE_1_HOST} --decode-port ${DECODE_1_PORT} --decode-rank ${DECODE_1_RANK} \

python  ./vllm/global_scheduler/client/create_comm_test.py \
    --prefill-host ${PREFILL_1_HOST} --prefill-port ${PREFILL_1_PORT} --prefill-rank ${PREFILL_1_RANK} \
    --decode-host ${DECODE_2_HOST} --decode-port ${DECODE_2_PORT} --decode-rank ${DECODE_2_RANK} \