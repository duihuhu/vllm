#!/bin/bash
echo "200 1090"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 1090

echo "190 43662"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 43662

echo "210 43700"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 43700


echo "150 46156"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 46156

echo "250 47714"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 47714


echo "100 48106"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 48106

echo "300 54903"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 54903


echo "50 55395"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 55395

echo "350 56273"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 56273


echo "1 69261"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 69261

echo "400 7329"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 1 --batch-size 4 --split-two-phase 1 --prompt-line 7329