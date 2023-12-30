#!/bin/bash
echo "1800 11092"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 11092

sleep 5
echo "1790 14248"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 14248

sleep 5
echo "1810 40421"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 40421
sleep 5

echo "1750 78766"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 78766
sleep 5
echo "1851 62264"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 62264
sleep 5

echo "1700 46147"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 46147
sleep 5
echo "1900 40233"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 40233
sleep 5

echo "1650 12434"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 12434
sleep 5
echo "1950 4768"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 4768

sleep 5
echo "1600 66231"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 66231
sleep 5
echo "2000 73059"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 73059
