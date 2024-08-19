import os
import time

lengths = [8, 16, 32]
i = 64 
while True:
    if i > 2048:
        break
    else:
        lengths.append(i)
        i += 64

base_command = "nsys profile \
    --wait=all \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --output=/home/jovyan/vllm/tests/kernels/profile_logs/log_{x}.txt \
    --force=true \
    --cudabacktrace=all \
    --cuda-memory-usage=true \
    --python-backtrace=cuda \
    python3 profile_xformers.py --num-tokens {x}"

base_command2 = "CUDA_VISIBLE_DEVICES=1,2,3,4 \
    ncu --metrics launch__block_count,launch__thread_count,duration,sm__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,l2__bytes_read.sum,l2__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --export temp \
    python3 /home/jovyan/vllm/benchmarks/benchmark_latency.py --input-len 4096 --tensor-parallel-size 1"

base_command3 = "ncu -i temp.ncu-rep --page details --csv --log-file temp.csv"

for length in lengths:
    command = base_command.format(x = length)
    os.system(command)
    time.sleep(3)