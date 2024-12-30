#!/bin/bash

PREFILL_HOST=10.156.154.242
PREFILL_PORT=8082
PREFILL_RANK="0"

DECODE_HOST=10.156.154.20
DECODE_PORT=8083
DECODE_RANK="1"
python  ./vllm/global_scheduler/client/create_comm_test.py \
    --prefill-host ${PREFILL_HOST} --prefill-port ${PREFILL_PORT} --prefill-rank ${PREFILL_RANK} \
    --decode-host ${DECODE_HOST} --decode-port ${DECODE_PORT} --decode-rank ${DECODE_RANK} \

