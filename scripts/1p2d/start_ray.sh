CUDA_VISIBLE_DEVICES=0,1 ray start --head --port=6379 --dashboard-port=8265
CUDA_VISIBLE_DEVICES=4,5 ray start --head --port=6381 --dashboard-port=8267