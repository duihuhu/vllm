import os

# 定义参数范围
input_lengths = [8]
iters = [1]
bs = [1,2,4,8,16,32,64,128,256,512]
#ratio = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#bs = [32, 64]

# 基础命令模板
base_command = "CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_latency.py --input-len {x} --num-seqs {y} --file-name /home/jovyan/vllm/benchmarks/logs/bd_{x}_{y}.txt"

# 遍历所有参数组合并生成命令

for ite in iters:
    for input_length in input_lengths:
        for b in bs:
            command = base_command.format(x=input_length,y=b)
            print(f"Executing: {command}")
            os.system(command)