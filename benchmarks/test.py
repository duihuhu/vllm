import os

# 定义参数范围
input_lengths = [64, 128, 256, 512, 1024]
output_lengths = [64, 128, 256, 512, 1024]
use_agg_block_options = [True, False]

# 基础命令模板
base_command = "python3 benchmark_latency.py --input-len {x} --output-len {y} --use-agg-block {z} --file /home/jovyan/hhy/vllm-hhy/benchmarks/log_{n}_{x}_{y}.txt"

# 遍历所有参数组合并生成命令
for x in input_lengths:
    for y in output_lengths:
        for z in use_agg_block_options:
            n = 1 if z else 0
            command = base_command.format(x=x, y=y, z=z, n=n)
            print(f"Executing: {command}")
            os.system(command)