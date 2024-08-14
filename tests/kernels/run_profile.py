import os
import time

lengths = [8, 16, 32]
i = 64 
while True:
    if i > 2048:
        break
    else:
        lengths.append(i)
        i += 64

base_command = "CUDA_VISIBLE_DEVICES=7 ncu python3 profile_xformers.py --num-tokens {x} > /home/jovyan/vllm/tests/kernels/log_{x}.txt"

for length in lengths:
    command = base_command.format(x = length)
    os.system(command)
    time.sleep(3)