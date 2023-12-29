#!/bin/bash
echo "1000 2339"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 2339

sleep 5
echo "990 19394"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 19394

sleep 5
echo "1010 5804"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 5804
sleep 5

echo "950 24423"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 24423
sleep 5
echo "1050 26298"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 26298
sleep 5

echo "900 27018"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 27018
sleep 5
echo "1100 34316"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 34316
sleep 5

echo "850 45861"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 45861
sleep 5
echo "1150 6539"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 6539

sleep 5
echo "800 10958"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 10958
sleep 5
echo "1200 86845"
python3 benchmark_throughput.py --backend vllm --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/ --tensor-parallel-size 4 --num-prompts 8 --batch-size 8 --split-two-phase 1 --prompt-line 86845
