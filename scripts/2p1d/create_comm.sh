#!/bin/bash

PREFILL_1_HOST=10.156.154.242
PREFILL_1_PORT=8082
PREFILL_1_RANK="0 1"

PREFILL_2_HOST=10.156.154.242
PREFILL_2_PORT=8084
PREFILL_2_RANK="2 3"


DECODE_HOST=10.156.154.20
DECODE_PORT=8083
DECODE_RANK="4 5"


python  ./vllm/global_scheduler/client/create_comm_test.py \
    --prefill-host ${PREFILL_1_HOST} --prefill-port ${PREFILL_1_PORT} --prefill-rank ${PREFILL_1_RANK} \
    --decode-host ${DECODE_HOST} --decode-port ${DECODE_PORT} --decode-rank ${DECODE_RANK} \

python  ./vllm/global_scheduler/client/create_comm_test.py \
    --prefill-host ${PREFILL_2_HOST} --prefill-port ${PREFILL_2_PORT} --prefill-rank ${PREFILL_2_RANK} \
    --decode-host ${DECODE_HOST} --decode-port ${DECODE_PORT} --decode-rank ${DECODE_RANK} \