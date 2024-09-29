 #!/bin/bash
python3 benchmark_latency_inference.py --input-len 18 --output-len 1 --num-iter 5 --num-light 1 --batch-size 2

sleep 5

python3 benchmark_latency_inference.py --input-len 18 --output-len 1 --num-iter 5 --num-light 2 --batch-size 2

sleep 5

python3 benchmark_latency_inference.py --input-len 18 --output-len 1 --num-iter 5 --num-light 4 --batch-size 4

sleep 5

python3 benchmark_latency_inference.py --input-len 18 --output-len 1 --num-iter 5 --num-light 8 --batch-size 8

sleep 5

python3 benchmark_latency_inference.py --input-len 18 --output-len 1 --num-iter 5 --num-light 16 --batch-size 16

sleep 5

python3 benchmark_latency_inference.py --input-len 18 --output-len 1 --num-iter 5 --num-light 32 --batch-size 32

sleep 5

python3 benchmark_latency_inference.py --input-len 18 --output-len 1 --num-iter 5 --num-light 64 --batch-size 64
