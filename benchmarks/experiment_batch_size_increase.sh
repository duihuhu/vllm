#!/bin/bash
#note: first need to ensure num_prompts
num_prompts=0
split_two_phase=0
for ((ts=1; ts<=4; ts=ts*4))
do
  for ((batch_size=2; batch_size<=4; batch_size=batch_size*2))
  do
    if [ $ts -eq 1 ]
    then
      num_prompts=2
    else
      num_prompts=4
    fi
    echo "current parameter: tensor-parallel-size batch_size num_prompts " $ts $batch_size  $num_prompts

    python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --num-prompts $num_prompts --tensor-parallel-size=$ts
    sleep 5
  done 
done