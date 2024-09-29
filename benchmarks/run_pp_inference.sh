 #!/bin/bash
echo "lplp"
python3 benchmark_latency_inference_lplp.py --input-len 18 --output-len 1 --num-iter 5 --num-light 1 --batch-size 2

sleep 5

python3 benchmark_latency_inference_lplp.py --input-len 18 --output-len 1 --num-iter 5 --num-light 2 --batch-size 2

sleep 5

python3 benchmark_latency_inference_lplp.py --input-len 18 --output-len 1 --num-iter 5 --num-light 4 --batch-size 4

sleep 5

python3 benchmark_latency_inference_lplp.py --input-len 18 --output-len 1 --num-iter 5 --num-light 8 --batch-size 8

sleep 5

python3 benchmark_latency_inference_lplp.py --input-len 18 --output-len 1 --num-iter 5 --num-light 16 --batch-size 16

sleep 5

python3 benchmark_latency_inference_lplp.py --input-len 18 --output-len 1 --num-iter 5 --num-light 32 --batch-size 32

sleep 5

python3 benchmark_latency_inference_lplp.py --input-len 18 --output-len 1 --num-iter 5 --num-light 64 --batch-size 64


echo "lphp"
python3 benchmark_latency_inference_lphp.py --input-len 18 --hinput-len 508 --output-len 1 --num-iter 5 --num-light 1 --num-heavy 1 --batch-size 2

sleep 5

python3 benchmark_latency_inference_lphp.py --input-len 18 --hinput-len 508 --output-len 1 --num-iter 5 --num-light 1 --num-heavy 3 --batch-size 4

echo "hplp"
python3 benchmark_latency_inference_lphp.py --input-len 18 --hinput-len 508 --output-len 1 --num-iter 5 --num-light 7 --num-heavy 1 --batch-size 8
sleep 5

python3 benchmark_latency_inference_lphp.py --input-len 18 --hinput-len 508 --output-len 1 --num-iter 5 --num-light 15 --num-heavy 1 --batch-size 16
sleep 5

python3 benchmark_latency_inference_lphp.py --input-len 18 --hinput-len 508 --output-len 1 --num-iter 5 --num-light 31 --num-heavy 1 --batch-size 32
sleep 5

python3 benchmark_latency_inference_lphp.py --input-len 18 --hinput-len 508 --output-len 1 --num-iter 5 --num-light 63 --num-heavy 1 --batch-size 64
sleep 5
