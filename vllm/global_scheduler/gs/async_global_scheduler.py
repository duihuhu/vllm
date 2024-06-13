import argparse
import json

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn
from vllm.global_scheduler.gs.global_meta import InstanceInfo, ReqCacheInfo, PrefixReqInfo, DistPolicy
from vllm.entrypoints.comm import EngineType
import vllm.global_scheduler.entrypoints_config as cfg
from typing import Dict, Set, List, AsyncGenerator, Tuple
import aiohttp
import random
from vllm.global_scheduler.gs.gs_radix_tree_manager import RadixTreeManager

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

#key: host_(service_port)_(machine_type)
#value: InstanceInfo 
instance_table: Dict[str, InstanceInfo] = {}

#record request and mprefill info
request_table: Dict[str, ReqCacheInfo] = {}

#record infight req id & decode instance, to use in d->p
infight_req: Dict[str, InstanceInfo] = ()

#record req id with its PrefixReqInfo
reqs_prefix_table: Dict[str, PrefixReqInfo] = {}

#record req id with mp\md information
req_engine_info: Dict[str, List[str]] = {}

coroutines: Dict[str, List] = {}

ep_rr_num = 0
ed_rr_num = 0
epd_rr_num = 0

block_size = 16

ep_token_tree = RadixTreeManager(block_size)
ed_token_tree = RadixTreeManager(block_size)
epd_token_tree = RadixTreeManager(block_size)


@app.post("/monitor_report")
async def monitor_report(request: Request) -> Response:
    headers = request.headers
    host = headers["host"]
    port = int(headers["port"])
    engine_type = headers["engine_type"]
    
    request_dict = await request.json()
    num_unfinished_requests = request_dict.pop("num_unfinished_requests")
    used_gpu_blocks = request_dict.pop("used_gpu_blocks")
    used_cpu_blocks = request_dict.pop("used_cpu_blocks")
    remained_gpu_blocks = request_dict.pop("remained_gpu_blocks")
    remained_cpu_blocks = request_dict.pop("remained_cpu_blocks") 
    global_ranks =  request_dict.pop("global_ranks") 
    timestamp = request_dict.pop("timestamp")
    key = host + "_" + str(port) + "_" + engine_type
    # print("key global_ranks", key, global_ranks)
    # print(key, unfinished_req, unfinished_tokens)
    if instance_table.get(key):
        instance = instance_table[key]
        instance.num_unfinished_reqs = num_unfinished_requests
        # instance['unfinished_tokens'] = unfinished_tokens
        instance.used_gpu_blocks = used_gpu_blocks
        instance.used_cpu_blocks = used_cpu_blocks
        instance.remained_gpu_blocks = remained_gpu_blocks
        instance.remained_cpu_blocks = remained_cpu_blocks
        instance.global_ranks = global_ranks
        instance.timestamp = timestamp
    else:
      instance = InstanceInfo(host, port, num_unfinished_requests, used_gpu_blocks,
                              used_cpu_blocks, remained_gpu_blocks, remained_cpu_blocks, EngineType(engine_type), timestamp, global_ranks)
      instance_table[key] = instance

    ret = {"result": 'monitor_report succ'}
    return ret

async def asyc_forward_request(request_dict, api_url, cdecode_host=None, cdecode_port=None, cdecode_ranks=None, cdecode_blocks=None):
    headers = {"User-Agent": "Test Client"}
    if cdecode_host:
        request_dict['cmeta_host'] = cdecode_host
        request_dict['cmeta_port'] = cdecode_port
        request_dict['cmeta_ranks'] = cdecode_ranks
        request_dict['cmeta_kv_len'] = cdecode_blocks
    # print("request info ", request_dict["request_id"], time.time())
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url=api_url, json=request_dict,
                                headers=headers) as response:
            if response.status == 200:
                delimiter=b"\0"
                buffer = b''  # 用于缓存数据块中的部分消息
                async for chunk in response.content.iter_any():
                    buffer += chunk  # 将新的数据块添加到缓冲区中
                    while delimiter in buffer:
                        index = buffer.index(delimiter)  # 查找分隔符在缓冲区中的位置
                        message = buffer[:index]  # 提取从缓冲区起始位置到分隔符位置的消息
                        yield message.strip()  # 返回提取的消息
                        buffer = buffer[index + len(delimiter):]  # 从缓冲区中移除已提取的消息和分隔符

async def asyc_forward_request_resp(request_dict, api_url):
    headers = {"User-Agent": "Test Client"}
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url=api_url, json=request_dict,
                                headers=headers) as response:
            return await response.text()

def select_instance(prompt_token_ids, policy, instance_type):
    policy = DistPolicy(policy)
    if policy == DistPolicy.RANDOM:
        return random_choice(instance_type)
    elif policy == DistPolicy.RR:
        return rr_choice(instance_type)
    elif policy == DistPolicy.PREFIX_CACHE:
        return prefix_cache_choice(prompt_token_ids, instance_type)
    elif policy == DistPolicy.LEAST_LOAD:
        return least_load_choice(instance_type)
    else:
        print("policy not finished ")
    
def select_disagg_instance(prompt_token_ids, prefill_policy, decode_policy) -> Tuple[InstanceInfo, InstanceInfo]:
    ep_instance = select_instance(prompt_token_ids, prefill_policy, EngineType.EPREFILL.value)
    ed_instance  = select_instance(prompt_token_ids, decode_policy, EngineType.EDECODE.value)
    return ep_instance, ed_instance

def select_agg_instance(prompt_token_ids, policy):
    epd_instance = select_instance(prompt_token_ids, policy, EngineType.EPD.value)
    return epd_instance

def random_instance(instance_type):
    instances = []
    for key, value in instance_table.items():
        if instance_type in key:
            instances.append(value)
    return random.choice(instances)

def random_choice(instance_type):
    instance = random_instance(instance_type)
    return instance

def rr_instance(instance_type):
    global ep_rr_num, ed_rr_num, epd_rr_num
    instances = []
    instance = None
    for key, value in instance_table.items():
        if instance_type in key:
            instances.append(value)
    if instance_type == EngineType.EPD.value:    
        instance = instances[epd_rr_num] 
        epd_rr_num = (epd_rr_num + 1) % len(instances)
    elif instance_type == EngineType.EPREFILL.value:
        instance = instances[ep_rr_num] 
        ep_rr_num = (ep_rr_num + 1) % len(instances)
    elif instance_type == EngineType.EDECODE.value:
        instance = instances[ed_rr_num] 
        ed_rr_num = (ed_rr_num + 1) % len(instances)
    return instance

def rr_choice(instance_type):
    instance = rr_instance(instance_type)
    return instance

def prefix_cache_instance(prompt_token_ids, instance_type):
    global ep_token_tree, ed_token_tree, epd_token_tree
    instances = None
    instance = None
    if instance_type == EngineType.EPREFILL.value:
        nodes = ep_token_tree.match(prompt_token_ids)
    elif instance_type == EngineType.EDECODE.value:
        nodes = ed_token_tree.match(prompt_token_ids)
    elif instance_type == EngineType.EPD.value:
        nodes = epd_token_tree.match(prompt_token_ids)
        
    #if not find, Degrade to other policy
    if not nodes:
        # instance = least_load_choice(instance_type=instance_type)
        instance = random_choice(instance_type=instance_type)
    else:
        start_instances = nodes[0].instances
        end_instances = nodes[-1].instances
        instances = list(set(start_instances) & set(end_instances))
        instance = random.choice(instances)
    return instance

def prefix_cache_choice(prompt_token_ids, instance_type):
    instance = prefix_cache_instance(prompt_token_ids, instance_type)    
    return instance

def least_load_instance(instance_type):
    instance = None
    least_load = 0
    for key, value in instance_table.items():
        if instance_type in key:
            if instance == None:
                instance = value
            else:
                if value.num_unfinished_reqs < least_load:
                    instance = value
                    least_load = value.num_unfinished_reqs
    return instance 

def least_load_choice(instance_type):
    instance = least_load_instance(instance_type)
    return instance

@app.post("/add_request")
async def add_request(request: Request) -> Response:
    request_dict = await request.json()   
    prompt_token_ids = request_dict["prompt_token_ids"]
    
    #TODO decide when use ep/ed and when use epd
    #select ep and ed instance for request 
    ep_instance, ed_instance = select_disagg_instance(prompt_token_ids, args.ep_policy, args.ed_policy)
    
    print("ep instance ", ep_instance.host, ep_instance.service_port)
    print("ed instance ", ed_instance.host, ed_instance.service_port)
    
    # select epd instance for request 
    # epd_instance = select_agg_instance(prompt_token_ids, args.epd_policy)

    #add prefill and decode info in request_dict, belong to one request
    request_dict["eprefill_host"] = ep_instance.host
    request_dict["eprefill_port"] = ep_instance.service_port
    request_dict["edecode_host"] = ed_instance.host
    request_dict["edecode_port"] = ed_instance.service_port
    prefill_response = asyc_forward_request(request_dict, cfg.forward_eprefill_url % 
                                                        (ep_instance.host, ep_instance.service_port))
    
    async def stream_results_prefill() -> AsyncGenerator[bytes, None]:
        async for resp in prefill_response:
            resp = resp.decode('utf-8')
            resp = json.loads(resp)
            #update gs prompt tree and decode tree
            if resp['n'] == 0 and args.ep_policy == "prefix":
                print("resp['prompt_token_ids'] ", resp['prompt_token_ids'])
                ep_token_tree.insert(resp['prompt_token_ids'], ep_instance)
                
            if resp['finished'] == True and args.ed_policy == "prefix":
                print("resp['prompt_token_ids'] + resp['prefilled_token_id'] ", resp['prompt_token_ids'])
                ed_token_tree.insert(resp['prompt_token_ids'] + resp['prefilled_token_id'], ed_instance)
                # epd_token_tree.insert(resp['prompt_token_ids'] + resp['prefilled_token_id'], epd_instance)
                
            yield (json.dumps(resp, ensure_ascii=False) + "\0").encode("utf-8")
    return StreamingResponse(stream_results_prefill())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", type=str, default="/workspace/opt-125m")
    parser.add_argument("--ep-policy",  type=str, default="random")
    parser.add_argument("--ed-policy",  type=str, default="random")
    parser.add_argument("--epd-policy",  type=str, default="random")


    args = parser.parse_args()
    uvicorn.run(app,
                host=cfg.global_scheduler_ip,
                port=cfg.global_scheduler_port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
