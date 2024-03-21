import argparse
import json

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import httpx
import random
from vllm.entrypoints.global_trie_tree import Trie
from vllm.entrypoints.global_meta import InstanceInfo, ReqCacheInfo, PrefixReqInfo, TransDataType
from vllm.entrypoints.comm import EngineType, RequestType
from vllm.transformers_utils.tokenizer import get_tokenizer
import vllm.entrypoints.entrypoints_config as cfg
from typing import Dict, Set, List
import asyncio
import time

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
tokenizer = None

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
    timestamp = request_dict.pop("timestamp")
    
    key = host + "_" + str(port) + "_" + engine_type
    # print(key, unfinished_req, unfinished_tokens)
    if instance_table.get(key):
        instance = instance_table[key]
        instance.num_unfinished_requests = num_unfinished_requests
        # instance['unfinished_tokens'] = unfinished_tokens
        instance.used_gpu_blocks = used_gpu_blocks
        instance.used_cpu_blocks = used_cpu_blocks
        instance.remained_gpu_blocks = remained_gpu_blocks
        instance.remained_cpu_blocks = remained_cpu_blocks
        instance.timestamp = timestamp
    else:
      instance = InstanceInfo(host, port, num_unfinished_requests, used_gpu_blocks,
                              used_cpu_blocks, remained_gpu_blocks, remained_cpu_blocks, EngineType[engine_type], timestamp)
      instance_table[key] = instance

    ret = {"result": 'monitor_report succ'}
    return ret

async def forward_request_to_server(request: Request, eprefill_forward_url, prefix_req=None) -> Response:
    # 获取原始请求的信息
    original_method = request.method
    original_headers = request.headers
    original_content = await request.body()
    if prefix_req:
        original_content_dict = json.loads(original_content)
        original_content_dict["prefix_req"] = prefix_req.__dict__
        original_content = json.dumps(original_content_dict)
    # 构建转发请求
    #use forward_response to response client
    #接受prefill first token
    #1) 返回给client
    #2) 选一个Decode
    #3) 转发给decode
    #4) 拿到decode的stream response，接收到两种类型结果
        #a) 响应
            # 响应并转发给P，主要是此时gs来选择之后，d通知p XX等准备完成）
        #b) 接收到增量token
            #1)返回给client
            #2)update tree？还是全部完成后update？
    #similar to server.py:62-79
    async with httpx.AsyncClient() as client:
        forward_response = await client.request(
            original_method,
            eprefill_forward_url,
            headers=dict(original_headers),
            content=original_content,
        )

    # 创建响应对象
    response = Response(
        content=forward_response.content,
        status_code=forward_response.status_code,
        headers=dict(forward_response.headers),
    )
    return response

async def forward_result_to_client(request: Request) -> Response:
    # 获取原始请求的信息
    original_method = request.method
    original_headers = request.headers
    original_content = await request.body()
    forward_url = cfg.forward_res_url % (cfg.client_ip, cfg.client_port)
    # 构建转发请求
    # forward_url = "http://example.com/destination_endpoint"
    async with httpx.AsyncClient() as client:
        forward_response = await client.request(
            original_method,
            forward_url,
            headers=dict(original_headers),
            content=original_content,
        )

    # 创建响应对象
    response = Response(
        content=forward_response.content,
        status_code=forward_response.status_code,
        headers=dict(forward_response.headers),
    )
    return response

@app.post('/add_results')
async def add_results(request: Request) -> Response:
    headers = request.headers
    engine_type = headers['engine_type']
    request_dict = await request.json()
    request_id = request_dict.get("request_id")
    output_text = request_dict.get("texts")
    finished = request_dict.get("finished")

    req_cache = request_table[request_id]
    if engine_type == EngineType.EPREFILL:
        req_cache.eprefill_host = headers["host"]
        req_cache.eprefill_port = headers["port"]
    elif engine_type == EngineType.EDECODE:
        req_cache.edecode_host = headers["host"]
        req_cache.edecode_port = headers["port"]  
    else:
        req_cache.epd_host = headers["host"]
        req_cache.epd_port = headers["port"]  
    
    req_cache.add_output(output_text[0])
    
    for prompt, output in zip(req_cache.unprefilled_prompts, req_cache.outputs):
        req_cache.add_token(tokenizer(prompt).input_ids)
        req_cache.add_token(tokenizer(output).input_ids)
    while req_cache.unprefilled_prompts:
        req_cache.add_prefilled_prompt(req_cache.unprefilled_prompts.pop(0))
    trie.insert(req_cache.token, req_cache.request_id)
    
    if finished:
        del coroutines[request_id]
    
    #forward result to client
    response = await forward_result_to_client(request)
    ret = {"result": 'add_results succ'}
    return ret

@app.post("/add_request")
async def add_request(request: Request) -> Response:
    request_dict = await request.json()
    request_id = request_dict.get("request_id")
    stream = request_dict.get("stream", False)
    prompt = request_dict.get("prompt")
    prompt_token = tokenizer(prompt).input_ids
    matched_req_ids, matched_len = trie.search(prompt_token)
    #no matched other req
    if not matched_req_ids:
        req_cache = ReqCacheInfo(request_id=request_id)
        req_cache.add_unprefilled_prompt(prompt)
        eprefill_url, eprefill_host, eprefill_port  = random_choose_mp()# or load balance_choose_mp
        req_cache.eprefill_host = eprefill_host
        req_cache.eprefill_port = eprefill_port
        
        mapping = [eprefill_host, eprefill_port, None, None]
        req_engine_info[request_id] = mapping
        
        #
        prefill_response = await forward_request_to_server(request, eprefill_url)
        
        #prefill_response to send to d
        #generate decode
        
        #response to p
        
        #response token to client
        
        #update trie tree by token or by seqs
        
        request_table[request_id] = req_cache
        infight_req.add(request_id)
    
    #todo when match 
    else:
        eprefill_url, eprefill_host, eprefill_port, edecode_host, edecode_port,  prefix_request_id, matched_machine = matched_reqs_choose_mp(matched_req_ids)
        if matched_machine:
            prefix_req_cache = request_table[prefix_request_id]
            prefix_req = PrefixReqInfo(request_id=prefix_request_id, type=EngineType.EPREFILL, matched_len=matched_len,
                                    mdecode_host=prefix_req_cache.eprefill_host, mdecode_port=prefix_req_cache.edecode_port, data_type=TransDataType.PART)
            reqs_prefix_table[request_id] = prefix_req
        else:
            prefix_req = PrefixReqInfo(request_id=prefix_request_id, type=EngineType.EDECODE, matched_len=matched_len,
                                mdecode_host=edecode_host, mdecode_port=edecode_port, data_type=TransDataType.FULL)
        
        mapping = [eprefill_host, eprefill_port, None, None]
        req_engine_info[request_id] = mapping
        
        req_cache = request_table[request_id]
        req_cache.eprefill_host = eprefill_host
        req_cache.eprefill_port = eprefill_port
        req_cache.add_token(prefix_req_cache.token[:matched_len])
        
        req_cache.add_unprefilled_prompt(prompt)
        response = await forward_request_to_server(request, eprefill_url, prefix_req)
        request_table[request_id] = req_cache
        infight_req.add(request_id)

    ret = {"text": 'add_request succ'}
    return ret

#todo find md by load, transfer to add_kv_request
#todo by and matched len
async def add_kv_request(request: Request) -> Response:
    request_dict = await request.json()
    request_id = request_dict.get("request_id")
    ed_url, ed_host, ed_port  = random_choose_md()
    mapping = req_engine_info[request_id]
    mapping[2] = ed_host
    mapping[3] = ed_port
    req_engine_info[request_id] = mapping
    response = await forward_request_to_server(request, ed_url)
    
    ret = {"result": 'add_kv_request succ'}
    return ret

#todo need keep request id and m_p_d host mapping
#transfer to p response_kv_prepared
async def response_kv_prepared(request: Request) -> Response:
    request_dict = await request.json()
    request_id = request_dict.get("request_id")
    machine = req_engine_info[request_id]
    ep_host, ep_port = machine[0], machine[1]
    res_url = cfg.forward_eprefill_res_url % (ep_host, ep_port)
    response = await forward_request_to_server(request, res_url)
    
    ret = {"result": 'response_kv_prepared succ'}
    return ret

def matched_reqs_choose_mp(matched_req_ids):
    #todo 
    # choose one of matched_req_ids, if req's mprefill is overload, choose random
    req_id = matched_req_ids[0]
    ep_info = request_table[req_id]
    eprefill_url = cfg.forward_eprefill_url % (ep_info.eprefill_host, ep_info.eprefill_port)
    matched_machine = True
    return eprefill_url, ep_info.eprefill_host, ep_info.eprefill_host, ep_info.edecode_host, ep_info.edecode_host, req_id, matched_machine

def random_choose_mpd():
    host = ""
    service_port = ""
    epd_instances = {key: instance for key, instance in instance_table.items() if instance.engine_type == EngineType.EPD}
    epd = random.sample(epd_instances.keys(), 1)[0]
    epd_instance = epd_instances[epd]
    host = epd_instances.host
    service_port = epd_instances.service_port
    url = cfg.forward_epd_url % (host, service_port)
    return url, host, service_port

#todo 
def cache_aware_mpd():
    return 

def random_choose_mp():
    host = ""
    service_port = ""
    ep_instances = {key: instance for key, instance in instance_table.items() if instance.machine_type == EngineType.EPREFILL}
    ep = random.sample(ep_instances.keys(), 1)[0]
    ep_instance = ep_instances[ep]
    host = ep_instance.host
    service_port = ep_instance.service_port
    url = cfg.forward_eprefill_url % (host, service_port)
    return url, host, service_port

def random_choose_md():
    host = ""
    service_port = ""
    ed_instances = {key: instance for key, instance in instance_table.items() if instance.machine_type == EngineType.EDECODE}
    ed = random.sample(ed_instances.keys(), 1)[0]
    ed_instance = ed_instances[ed]
    host = ed_instance.host
    service_port = ed_instance.service_port
    url = cfg.forward_edecode_url % (host, service_port)
    return url, host, service_port
#todo 
def cache_aware_mp():
    return

#todo 
def load_balacne_mp():
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="/workspace/opt-125m")

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = get_tokenizer(args.tokenizer)
    trie = Trie()
    uvicorn.run(app,
                host=cfg.host_ip,
                port=cfg.global_scheduler_port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
