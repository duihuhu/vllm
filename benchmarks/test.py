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
tps = [2,4]

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=1,2,3,4 python3 benchmark_latency.py --input-len {x} --tensor-parallel-size {y} --file-name /home/jovyan/vllm/benchmarks/tp_{y}/temp_{x}.txt"

# 遍历所有参数组合并生成命令
for tp in tps:
    for input_length in input_lengths:
        command = base_command.format(x=input_length, y=tp)
        print(f"Executing: {command}")
        os.system(command)
        time.sleep(3)