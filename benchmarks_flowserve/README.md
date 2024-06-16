# Benchmark FlowServe

## Launch Global Scheduler
```bash
python3 ./vllm/global_scheduler/async_global_scheduler_notree_nocontrol.py --model /data/zhaoyiyang/Llama-2-7B-fp16/
```

## Launch Prefill instances
```bash
python3 ./vllm/entrypoints/server.py  --local_host 127.0.0.1 --model=/data/zhaoyiyang/Llama-2-7B-fp16/ --local_port 8082 --worker-use-ray  --tensor-parallel-size 2 --block-size 16 --enable-separate --role=prompt --enable-direct --enable-layer --enable-dcache --enable-radix-caching 
```

## Launch Decode instances
```bash
python3 ./vllm/entrypoints/server.py  --local_host 127.0.0.1 --model=/data/zhaoyiyang/Llama-2-7B-fp16/ --local_port 8083 --worker-use-ray  --tensor-parallel-size 2 --block-size 16 --enable-separate --role=decoder --enable-direct  --enable-layer  --enable-dcache --enable-radix-caching 
```

## Create communication domain

```bash
python3  vllm/global_scheduler/client/create_comm_test.py
```
Before running the preceding command, take a look at the last few lines of create_comm_test.py file.

If the Prefill instance (P) has port 8082 and uses GPUs 0 and 1 for tensor parallelism, and the Decode instance (D) has port 8083 and uses GPUs 2 and 3 for tensor parallelism, the following code from create_comm_test.py can be used to create a communication domain from P to D (or optionally from D to P).

```python
resp = create_comm(8082,[0,1],8083,[2,3], "sender")
# resp = create_comm(8082,[0,1],8083,[2,3], "recv")
```

## Launch Client
```bash
python3 ./benchmarks_flowserve/main.py --dataset LooGLE --request-rate 12.8 --num-requests 64
```

## Options for FlowServe services

- --enable-separate: PD-disaggregated
- --enable-layer: Layer-level transfer (request-level transfer by default)
- --enable-dcache: Need to enable the comminucation domain from D to P.
- --enable-radix-caching: Enable radix caching