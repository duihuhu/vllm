import os
import time

# 定义参数范围
input_lengths = [8,16,32]
i = 64
while True:
    if i > 2048:
        break
    else:
        input_lengths.append(i)
        i += 64

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=4,5,6,7 python3 benchmark_latency.py --input-len {x} --file-name /home/jovyan/vllm/benchmarks/profile_logs/tp2_preattnnorm_{x}.txt"

# 遍历所有参数组合并生成命令
for input_length in input_lengths:
    command = base_command.format(x = input_length)
    print(f"Executing: {command}")
    os.system(command)