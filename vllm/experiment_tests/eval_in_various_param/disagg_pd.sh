#!/bin/bash
dirname="e2e_exp_data"
if [ -d "$dirname" ]; then
    rm -rf ${dirname}
    if [ $? -eq 0 ]; then
        mkdir ${dirname}
        if [ $? -eq 0 ]; then
            echo "Directory ${dirname} created successfully."
        else
            echo "Error creating directory ${dirname}."
        fi
    else
        echo "Error deleting directory ${dirname}."
    fi
else
    mkdir  -p ${dirname}
    if [ $? -eq 0 ]; then
        echo "Directory ${dirname} created successfully."
    else
        echo "Error creating directory ${dirname}."
    fi
fi
name="disagg"
input_lens=(64 128 256 512 1024 2048)
output_lens=(16 32)
request_rates=(3.2 6.4 12.8 25.6 51.2 102.4)
num_requests=256
mkdir ${dirname}
for input_len in "${input_lens[@]}"; do
  for output_len in "${output_lens[@]}"; do
    for request_rate in "${request_rates[@]}"; do
        output_file="${dirname}/${name}_${input_len}_${output_len}_${request_rate}_${num_requests}.txt"
        command="python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len $input_len --output-len $output_len --request-rate $request_rate --num-requests $num_requests"
        eval "$command" > "$output_file"
        echo "Sleeping for 10 seconds..."
        sleep 10
    done
  done
done