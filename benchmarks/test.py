import os

# 定义参数范围
input_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4090]
ratio = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#output_lengths = [64, 128, 256, 512, 1024]
#use_agg_block_options = [True, False]

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_latency.py --input-len {x} --reuse-ratio {y} --enable-radix-caching --use-agg-block --file-path /home/jovyan/hhy/vllm-hhy/benchmarks/log2_{x}_{y}.txt"

# 遍历所有参数组合并生成命令
for x in input_lengths:
    for y in ratio:
        #for z in use_agg_block_options:
            #n = 1 if z else 0
        command = base_command.format(x=x, y=y)
        print(f"Executing: {command}")
        os.system(command)