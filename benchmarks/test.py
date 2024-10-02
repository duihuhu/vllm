import os
import time

# 定义参数范围
input_lengths = []
i = 2048
while True:
    if i > 4096:
        break
    else:
        input_lengths.append(i)
        i += 64

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 benchmark_latency.py --input-len {x} --file-name /home/jovyan/vllm/benchmarks/profile_logs_tp1_long_range/tp1_preattnnorm_{x}.txt"

# 遍历所有参数组合并生成命令
for input_length in input_lengths:
    command = base_command.format(x = input_length)
    print(f"Executing: {command}")
    os.system(command)
    #time.sleep(3)