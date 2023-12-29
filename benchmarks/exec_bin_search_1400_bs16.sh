#!/bin/bash
echo "1400 85392"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 85392

sleep 5
echo "1390 28091"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 28091

sleep 5
echo "1410 10364"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 10364
sleep 5

echo "1350 1989"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 1989
sleep 5
echo "1450 68490"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 68490
sleep 5

echo "1300 8264"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 8264
sleep 5
echo "1500 20543"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 20543
sleep 5

echo "1250 15860"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 15860
sleep 5
echo "1550 58061"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 58061

sleep 5
echo "1200 86845"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 86845
sleep 5
echo "1600 66231"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 16 --batch-size 16 --split-two-phase 1 --prompt-line 66231
