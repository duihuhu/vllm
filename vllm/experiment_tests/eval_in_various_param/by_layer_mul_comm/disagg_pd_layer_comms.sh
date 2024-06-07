#!/bin/bash
dirname="e2e_exp"
type="disagg_pd_layer_comms"
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

input_lens=(64 128 256 512 1024 2048)
output_lens=(16 32)
request_rates=(3.2 6.4 12.8 25.6 51.2 102.4)
num_requests=256
for input_len in "${input_lens[@]}"; do
  for output_len in "${output_lens[@]}"; do
    for request_rate in "${request_rates[@]}"; do
        output_file="${dirtype}/${type}_${input_len}_${output_len}_${request_rate}_${num_requests}.txt"
        command="python3 ./vllm/global_scheduler/client/api_client_async_req_rate_len.py --input-len $input_len --output-len $output_len --request-rate $request_rate --num-requests $num_requests"
        eval "$command" > "$output_file"
        echo "Sleeping for 10 seconds..."
        sleep 10
    done
  done
done