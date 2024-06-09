import argparse
import json

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from vllm.global_scheduler.global_radix_tree import RadixCache
from vllm.global_scheduler.global_meta import InstanceInfo, ReqCacheInfo, PrefixReqInfo, TransDataType
from vllm.entrypoints.comm import EngineType
from vllm.transformers_utils.tokenizer import get_tokenizer
import vllm.global_scheduler.entrypoints_config as cfg
from typing import Dict, Set, List, Optional, AsyncGenerator
import asyncio
import aiohttp
import requests
import random

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
tokenizer = None

#config instance:gs first to read instance.json to know how many instance 
config_instance_table: Dict[str, InstanceInfo] = {}
num_comm = 0

#key: host_(service_port)_(machine_type)
#value: InstanceInfo 
instance_table: Dict[str, InstanceInfo] = {}
#record request and mprefill info
request_table: Dict[str, ReqCacheInfo] = {}
#record infight req id
infight_req: Set[str] = ()
#record req id with its PrefixReqInfo
reqs_prefix_table: Dict[str, PrefixReqInfo] = {}
#record req id with mp\md information
req_engine_info: Dict[str, List[str]] = {}

coroutines: Dict[str, List] = {}

# gs_ptoken_tree = RadixCache()
# gs_dtoken_tree = RadixCache()

def post_request(api_url, request_dict: Optional[Dict] = {}):
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=request_dict)
    return response

#to init comm between instances
def create_comm(src_instance: InstanceInfo, dst_instance: InstanceInfo):
    uniqe_id_api_url = cfg.comm_uniqe_id_url % (src_instance.host, src_instance.service_port)
    dst_channel = "_".join([str(rank) for rank in dst_instance.global_ranks])
    response = post_request(uniqe_id_api_url, {"dst_channel": dst_channel})

    # create_comm_url = cfg.create_comm_url % (dst_instance.host, dst_instance.service_port)
    # response = post_request(create_comm_url, response.json())

@app.post("/init_comm")
async def init_comm(request: Request) -> Response:
    config_instances = [config_instance_table.values()]
    instances: List[InstanceInfo] = [instance_table.values()]
    if len(instances) == config_instances:
        for src_comm in range(len(instances)):
            for dst_comm in range(src_comm + 1, len(instances)):
                for num in range(num_comm):
                    create_comm(instances[src_comm], instances[dst_comm])
            

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
        instance.num_unfinished_requests = num_unfinished_requests
        # instance['unfinished_tokens'] = unfinished_tokens
        instance.used_gpu_blocks = used_gpu_blocks
        instance.used_cpu_blocks = used_cpu_blocks
        instance.remained_gpu_blocks = remained_gpu_blocks
        instance.remained_cpu_blocks = remained_cpu_blocks
        instance.global_ranks = global_ranks
        instance.timestamp = timestamp
    else:
      instance = InstanceInfo(host, port, num_unfinished_requests, used_gpu_blocks,
                              used_cpu_blocks, remained_gpu_blocks, remained_cpu_blocks, EngineType[engine_type], timestamp, global_ranks)
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
        
def search_prefix(radix_tree, token_ids):
    value, node, last_node_matched_len = radix_tree.only_match_prefix(tuple(token_ids))
    if value:
        return True, value, node.node_addr[0], last_node_matched_len
    else:
        return False, [], [], 0

def get_epd_cached_meta(ptree, dtree, token_ids):
    ep_host = None
    ep_port = None
    cd_host = None
    cd_port = None
    cd_ranks = None
    cd_blocks = 0
    ed_host = None
    ed_port = None
    p_matched, p_tokens, p_node, p_last_node_matched_len = search_prefix(ptree, token_ids)
    if p_matched:
        ep_host, ep_port = p_node.split("_")
    else:
        ep_host, ep_port = cfg.eprefill_host, cfg.eprefill_port
    d_matched, d_tokens, d_node, d_last_node_matched_len = search_prefix(dtree, token_ids)
    if d_matched:
        cd_host, cd_port = d_node.split("_")
        # print("d_node ", d_node)
        # print("d_tokens ", d_tokens, len(d_tokens), len(p_tokens))
        instance = instance_table.get(d_node + "_" + cfg.edecode_label)
        cd_ranks = instance.global_ranks
        cd_blocks = len(d_tokens)
    else:
        cd_host, cd_port = None, None
    ed_host = cfg.edecode_host
    ed_port = cfg.edecode_port
    return ep_host, ep_port, cd_host, cd_port, cd_ranks, ed_host, ed_port, cd_blocks

@app.post("/add_request")
async def add_request(request: Request) -> Response:
    request_dict = await request.json()   
    prompt_token_ids = request_dict["prompt_token_ids"]

    eprefill_host, eprefill_port, cdecode_host, cdecode_port, cdecode_ranks,\
        edecode_host, edecode_port, cdecode_blocks  = cfg.eprefill_host, cfg.eprefill_port, None, None, None, cfg.edecode_host, cfg.edecode_port, None
        
    eprefill_port = random.choice([cfg.eprefill_port])
    edecode_port = random.choice([cfg.edecode_port])
    request_dict["edecode_host"] = edecode_host
    request_dict["edecode_port"] = edecode_port
    prefill_response = asyc_forward_request(request_dict, cfg.forward_eprefill_url % 
                                                        (eprefill_host, eprefill_port), cdecode_host, cdecode_port, cdecode_ranks, cdecode_blocks)
    
    async def stream_results_prefill() -> AsyncGenerator[bytes, None]:
        async for resp in prefill_response:
            resp = resp.decode('utf-8')
            resp = json.loads(resp)
            yield (json.dumps(resp, ensure_ascii=False) + "\0").encode("utf-8")
    return StreamingResponse(stream_results_prefill())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="/workspace/opt-125m")
    parser.add_argument("--enable-dcache",  action="store_true", help=('enable pass decode to prefill cache '))
    parser.add_argument("--enable-separate",  action="store_true")
    parser.add_argument("--enable-layer",  action="store_true")

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = get_tokenizer(args.tokenizer)
    uvicorn.run(app,
                host=cfg.global_scheduler_ip,
                port=cfg.global_scheduler_port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
