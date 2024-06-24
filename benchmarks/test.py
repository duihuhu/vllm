import os

# 定义参数范围
input_lengths = [1024,2048,4080]
ratio = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bs = [32, 64]

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_latency.py --input-len {x} --block-size {y} --ratio {z} --enable-radix-caching --file-name /home/jovyan/hhy/vllm-hhy/benchmarks/logs6/log_{y}_{x}_{z}.txt"

# 遍历所有参数组合并生成命令

for y in bs:
    for x in input_lengths:
        for z in ratio:
            command = base_command.format(x=x,y=y,z=z)
            print(f"Executing: {command}")
            os.system(command)