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

base_command2 = "CUDA_VISIBLE_DEVICES=5,6,7,8 \
    ncu --max-waves 5 \
    --metrics launch__thread_count,duration,sm__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on.avg.pct_of_peak_sustained_elapsed,dram__sectors_read.sum,dram__sectors_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --call-stack \
    --nvtx \
    --export temp10 \
    python3 /home/jovyan/vllm/benchmarks/benchmark_latency.py \
    --model /home/jovyan/models/Llama-2-13b-hf/ \
    --tensor-parallel-size {x} \
    --input-len {y} \
    --output-len 1 \
    --num-seqs 1 \
    --batch-size 1 \
    --num-iters 1"

base_command3 = "ncu -i temp.ncu-rep --page details --csv --log-file temp.csv"

for length in lengths:
    command = base_command2.format(x = 1, y = length)
    os.system(command)
    time.sleep(3)