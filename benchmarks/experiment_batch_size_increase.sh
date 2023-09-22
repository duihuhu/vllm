#!/bin/bash
num_prompts=0
for ((batch_size=2; batch_size<=64; batch_size=batch_size*2))
do
  for ((ts=1; ts<=4; ts=ts*4))
  do
    if [ $ts -eq 1 ]
    then
      $num_prompts=64
    else
      $num_prompts=128
    fi
    echo $batch_size $ts $num_prompts

    # python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json \
    # --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --num-prompts 64 --tensor-parallel-size=1

  done 
done