 #!/bin/bash
python3 benchmark_latency.py --input-len 18 --output-len 1 --num-iter 5 --n 1 --batch-size 2

sleep 5

python3 benchmark_latency.py --input-len 18 --output-len 1 --num-iter 5 --n 2 --batch-size 2

sleep 5


python3 benchmark_latency.py --input-len 18 --output-len 1 --num-iter 5 --n 4 --batch-size 4

sleep 5

python3 benchmark_latency.py --input-len 18 --output-len 1 --num-iter 5 --n 8 --batch-size 8

sleep 5

python3 benchmark_latency.py --input-len 18 --output-len 1 --num-iter 5 --n 16 --batch-size 16

sleep 5

python3 benchmark_latency.py --input-len 18 --output-len 1 --num-iter 5 --n 32 --batch-size 32

sleep 5

python3 benchmark_latency.py --input-len 18 --output-len 1 --num-iter 5 --n 64 --batch-size 64
