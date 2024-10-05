import os
import time

# 定义参数范围
input_lengths = []
i = 16
while True:
    if i > 4096:
        break
    else:
        input_lengths.append(i)
        i += 16

tps = [1, 2]

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 benchmark_latency.py --input-len {x} --tensor-parallel-size {y} --file-name /home/jovyan/vllm/benchmarks/test_logs/tp_{y}_{x}.txt"

# 遍历所有参数组合并生成命令
for tp in tps:
    for input_length in input_lengths:
        command = base_command.format(x = input_length, y = tp)
        print(f"Executing: {command}")
        os.system(command)