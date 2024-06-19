一、手动指定preill/decode实例：
测试及启动步骤：
1）启动prefill
P侧：
python3 ./vllm/entrypoints/server.py  --local_host 127.0.0.1 --model=/home/jovyan/models/Llama-2-13b-hf/ --local_port 8082 --worker-use-ray  --tensor-parallel-size 2 --block-size 16 --enable-separate --role=prompt --enable-direct --enable-layer --enable-dcache --enable-radix-caching 

2）启动decoder
D侧
python3 ./vllm/entrypoints/server.py  --local_host 127.0.0.1 --model=/home/jovyan/models/Llama-2-13b-hf/ --local_port 8084 --worker-use-ray  --tensor-parallel-size 2 --block-size 16 --enable-separate --role=decoder --enable-direct  --enable-layer  --enable-dcache --enable-radix-caching 

3）创建通信域(与卡对应，这个需要手动触发，如prefill启动第一个实例为0,1；参见示例）
python3  vllm/global_scheduler/client/create_comm_test.py

.4）启动全局调度器
global scheduler
python3 ./vllm/global_scheduler/async_global_scheduler_notree_nocontrol.py --model /home/jovyan/models/Llama-2-13b-hf/

5）发负载（具体批量测试和画图脚本参见experiment_tests）
client
python3 ./vllm/global_scheduler/client/api_client_async_req_rate_len.py --input-len 32 --output-len 32 --request-rate 12.8 --num-requests 64

分离的阶段测试中的必选项：
--enable-separate（分离必须项，不加是合部）
--role=prompt（分离必选项）

分离的阶段测试中的非必选项：
--enable-layer：layer级别传输（默认是req粒度, layer的话可以忽略了应该）
--enable-dcache：开启d-> p回传
--enable-radix-caching: 开启radix缓存（目前只有hbm之间的）
--enable-trans-to-dram: 缓存回传和传输，是否会进入dram
--use-agg-block:是否使用优化后的mm布局
--enable-radix-evictor:是否开启缓存驱逐，开启的话，达到阈值时候，对满足条件的gpu缓存进入换出驱逐，对cpu缓存进行直接free
-------------------------------------------------------------------------------------------------------------

二、gs自动选择preill/decode实例：

测试及启动步骤：
1）启动全局调度器
global scheduler
python3 ./vllm/global_scheduler/gs/async_global_scheduler.py --model /home/jovyan/models/Llama-2-13b-hf/ --ep-policy rr --ed-policy rr
其中：
--ep-policy / --ed-policy: random, rr, prefix, least

2）启动prefill
P侧：
python3 ./vllm/entrypoints/server.py  --local_host 127.0.0.1 --model=/home/jovyan/models/Llama-2-13b-hf/ --local_port 8082 --worker-use-ray  --tensor-parallel-size 2 --block-size 16 --enable-separate --role=prompt --enable-direct --enable-layer --enable-dcache --enable-radix-caching 

3）启动decoder
D侧
python3 ./vllm/entrypoints/server.py  --local_host 127.0.0.1 --model=/home/jovyan/models/Llama-2-13b-hf/ --local_port 8084 --worker-use-ray  --tensor-parallel-size 2 --block-size 16 --enable-separate --role=decoder --enable-direct  --enable-layer  --enable-dcache --enable-radix-caching 

4）创建通信域(与卡对应，这个需要手动触发，如prefill启动第一个实例为0,1；参见示例）
python3  vllm/global_scheduler/client/create_comm_test.py

5）发负载（具体批量测试和画图脚本参见experiment_tests）
client
python3 ./vllm/global_scheduler/client/api_client_async_req_rate_len.py --input-len 32 --output-len 32 --request-rate 12.8 --num-requests 64

分离的阶段测试中的必选项：
--enable-separate（分离必须项，不加是合部）
--role=prompt（分离必选项）

分离的阶段测试中的非必选项：
--enable-layer：layer级别传输（默认是req粒度）
--enable-dcache：开启d-> p回传
--enable-radix-caching: 开启radix缓存（目前只有hbm之间的）