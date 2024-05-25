#!/bin/bash

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 64 --output-len 16 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 64 --output-len 8 --request-rate 1.6 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 64 --output-len 16 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 64 --output-len 8 --request-rate 3.2 --num-requests 256
sleep 10


python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 64 --output-len 16 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 64 --output-len 8 --request-rate 6.4 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 32 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 16 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 8 --request-rate 1.6 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 32 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 16 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 8 --request-rate 3.2 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 32 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 16 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 128 --output-len 8 --request-rate 6.4 --num-requests 256
sleep 10


python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 64 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 32 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 16 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 8 --request-rate 1.6 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 64 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 32 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 16 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 8 --request-rate 3.2 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 64 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 32 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 16 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 256 --output-len 8 --request-rate 6.4 --num-requests 256
sleep 10


python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 128 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 64 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 32 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 16 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 8 --request-rate 1.6 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 128 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 64 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 32 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 16 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 8 --request-rate 3.2 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 128 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 64 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 32 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 16 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 512 --output-len 8 --request-rate 6.4 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 256 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 128 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 64 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 32 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 16 --request-rate 1.6 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 256 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 128 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 64 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 32 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 16 --request-rate 3.2 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 256 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 128 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 64 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 32 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 1024 --output-len 16 --request-rate 6.4 --num-requests 256
sleep 10


python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 512 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 256 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 128 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 64 --request-rate 1.6 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 32 --request-rate 1.6 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 512 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 256 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 128 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 64 --request-rate 3.2 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 32 --request-rate 3.2 --num-requests 256
sleep 10

python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 512 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 256 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 128 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 64 --request-rate 6.4 --num-requests 256
sleep 10
python3 ./vllm/entrypoints/api_client_async_req_rate_len.py --input-len 2048 --output-len 32 --request-rate 6.4 --num-requests 256
sleep 10