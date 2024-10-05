import os
import time

# 定义参数范围
input_lengths = []
i = 64
while True:
    if i > 4096:
        break
    else:
        input_lengths.append(i)
        i += 64

ratios = [25, 50, 75]

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 benchmark_latency.py --input-len {x} --ratio {y} --enable-radix-caching --file-name /home/jovyan/vllm/benchmarks/test_logs/ratio_{y}_{x}.txt"

# 遍历所有参数组合并生成命令
for r in ratios:
    for input_length in input_lengths:
        command = base_command.format(x = input_length, y = r)
        print(f"Executing: {command}")
        os.system(command)