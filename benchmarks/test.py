import os

# 定义参数范围
input_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4080]
#ratio = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#bs = [2, 4]

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_latency.py --input-len {x} --file-name /home/jovyan/hhy/vllm-hhy/benchmarks/log2b_{x}.txt"

# 遍历所有参数组合并生成命令
for x in input_lengths:
    #for y in ratio:
    command = base_command.format(x=x)
    print(f"Executing: {command}")
    os.system(command)