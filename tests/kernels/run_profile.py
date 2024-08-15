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

base_command = "nsys profile \
    --wait=all \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --output=/home/jovyan/vllm/tests/kernels/profile_logs/log_{x}.txt \
    --force=true \
    --cudabacktrace=all \
    --cuda-memory-usage=true \
    --python-backtrace=cuda \
    python3 profile_xformers.py --num-tokens {x}"

for length in lengths:
    command = base_command.format(x = length)
    os.system(command)
    time.sleep(3)