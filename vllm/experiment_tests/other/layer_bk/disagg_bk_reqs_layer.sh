#!/bin/bash
dirname="e2e_breakdown"
type="disagg_layer"
dirtype="${dirname}/${type}"
if [ -d "$dirtype" ]; then
    rm -rf ${dirtype}
    if [ $? -eq 0 ]; then
        mkdir ${dirtype}
        if [ $? -eq 0 ]; then
            echo "Directory ${dirtype} created successfully."
        else
            echo "Error creating directory ${dirtype}."
        fi
    else
        echo "Error deleting directory ${dirtype}."
    fi
else
    mkdir  -p ${dirtype}
    if [ $? -eq 0 ]; then
        echo "Directory ${dirtype} created successfully."
    else
        echo "Error creating directory ${dirtype}."
    fi
fi

input_lens=(64 128 256 512 1024)
output_lens=(16 32)
request_rates=(3.2 6.4 12.8 25.6 51.2 102.4)
num_requests=256
for input_len in "${input_lens[@]}"; do
  for output_len in "${output_lens[@]}"; do
    for request_rate in "${request_rates[@]}"; do
        output_file="${dirtype}/${type}_${input_len}_${output_len}_${request_rate}_${num_requests}.txt"
        command="python3 ./vllm/global_scheduler/client/api_client_async_req_rate_len.py --input-len $input_len --output-len $output_len --request-rate $request_rate --num-requests $num_requests"
        eval "$command"
        sleep 2
        mv /home/jovyan/vllm/prefill_add_kv_request_layer.txt ${dirtype}/${type}_${input_len}_${output_len}_${request_rate}_${num_requests}_prefill_add_kv_request_layer.txt

        mv /home/jovyan/vllm/prefill_send_query_kv_to_decode_layer.txt ${dirtype}/${type}_${input_len}_${output_len}_${request_rate}_${num_requests}_prefill_send_query_kv_to_decode_layer.txt
        
        mv /home/jovyan/vllm/decode_add_request_to_running_layer.txt ${dirtype}/${type}_${input_len}_${output_len}_${request_rate}_${num_requests}_decode_add_request_to_running_layer.txt

        mv /home/jovyan/vllm/decode_finished_reqs_layer.txt ${dirtype}/${type}_${input_len}_${output_len}_${request_rate}_${num_requests}_decode_finished_reqs_layer.txt
        sleep 10
    done
  done
done