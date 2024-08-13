import os

# 定义参数范围
input_lengths = [8,16,32,64,128,256,384,512,640,768,896,1024,1152,1280]

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_latency.py --input-len {x} --enforce-eager --file-name /home/jovyan/vllm/benchmarks/temp_{x}.txt"

# 遍历所有参数组合并生成命令

for input_length in input_lengths:
    command = base_command.format(x=input_length)
    print(f"Executing: {command}")
    os.system(command)