#!/bin/bash 
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 1 --turns 4 --stream > log4_r1_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 1 --turns 6 --stream > log6_r1_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 1 --turns 8 --stream > log8_r1_cache.txt
sleep 10

python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 2 --turns 4 --stream > log4_r2_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 2 --turns 6 --stream > log6_r2_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 2 --turns 8 --stream > log8_r2_cache.txt
sleep 10

python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 3 --turns 4 --stream > log4_r3_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 3 --turns 6 --stream > log6_r3_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 3 --turns 8 --stream > log8_r3_cache.txt
sleep 10

python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 4 --turns 4 --stream > log4_r4_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 4 --turns 6 --stream > log6_r4_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 4 --turns 8 --stream > log8_r4_cache.txt
sleep 10

python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 5 --turns 4 --stream > log4_r5_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 5 --turns 6 --stream > log6_r5_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 5 --turns 8 --stream > log8_r5_cache.txt
sleep 10


python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 6 --turns 4 --stream > log4_r5_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 6 --turns 6 --stream > log6_r5_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 6 --turns 8 --stream > log8_r5_cache.txt
sleep 10


python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 7 --turns 4 --stream > log4_r7_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 7 --turns 6 --stream > log6_r7_cache.txt
sleep 10

python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 8 --turns 4 --stream > log4_r8_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 8 --turns 6 --stream > log6_r8_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 8 --turns 8 --stream > log8_r8_cache.txt
sleep 10

python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 9 --turns 4 --stream > log4_r9_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 9 --turns 6 --stream > log6_r9_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 9 --turns 8 --stream > log8_r9_cache.txt
sleep 10


python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 10 --turns 4 --stream > log4_r10_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 10 --turns 6 --stream > log6_r10_cache.txt
sleep 10
python3 ./vllm/entrypoints/api_client_mul_conversation_threading.py --session 100 --request-rate 10 --turns 8 --stream > log8_r10_cache.txt
sleep 10