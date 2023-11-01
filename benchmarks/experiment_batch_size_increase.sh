#!/bin/bash
#note: first need to ensure num_prompts, batch_size
num_prompts=32
split_two_phase=1 #0 is not split , 1 is split
for ((ts=2; ts<=2; ts=ts*2))
do
  for ((batch_size=2; batch_size<=32; batch_size=batch_size*2))
  do
    if [ $ts -eq 1 ]
    then
      num_prompts=32
    else
      num_prompts=32
    fi
    echo "current parameter: tensor-parallel-size batch_size num_prompts " $ts $batch_size  $num_prompts

    python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ \
    --num-prompts $num_prompts --tensor-parallel-size=$ts --split-two-phase=$split_two_phase
    sleep 5
  done 
done