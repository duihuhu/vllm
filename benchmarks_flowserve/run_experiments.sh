#!/bin/bash

# Configurable parameters
basename="end2end_exp_results"
dataset="ReAct"
type="disagg_layer"
request_rates=(3.2 6.4 12.8 25.6 51.2 102.4)
num_requests=256

# Derived parameters
dirname="${basename}/${dataset}/${type}"

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
    mkdir -p ${dirname}
    if [ $? -eq 0 ]; then
        echo "Directory ${dirname} created successfully."
    else
        echo "Error creating directory ${dirname}."
    fi
fi

for request_rate in "${request_rates[@]}"; do
    output_file="${dirname}/${dataset}_${type}_${request_rate}_${num_requests}.csv"
    command="python3 ./vllm/global_scheduler/client/api_client_async_req_rate_len.py --dataset ${dataset} --request-rate $request_rate --num-requests $num_requests"
    eval "$command" > "$output_file"
    echo "Sleeping for 10 seconds..."
    sleep 10
done