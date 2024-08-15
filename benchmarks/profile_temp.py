import torch
import torch.profiler
import os

# 设置 profiler
with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True) as prof:
    
    base_command = "CUDA_VISIBLE_DEVICES=1,2,3,4 python3 benchmark_latency.py \
        --tensor-parallel-size 1 \
        --input-len 4096 \
        --output-len 1\
        --num-seqs 1 \
        --batch-size 1 \
        --num-iters 30"
    
    os.system(base_command)

# 遍历事件，获取每个 kernel 的启动次数和线程块信息
for event in prof.profiler.events():
    if event.device_type == torch.device('cuda'):
        print(f"Kernel: {event.name}")
        print(f"Launch Count: {event.count}")
        print(f"Thread Blocks: {event.thread_blocks}")