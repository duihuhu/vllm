import argparse
import json

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import httpx
import api_global_scheduer_config as cfg
import random
from api_global_trie_tree import Trie
from api_global_meta import MachineType, InstanceInfo, ReqCacheInfo, PrefixReqInfo, TransDataType
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import Dict, Set, List

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

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
req_machine_info: Dict[str, List[str]] = {}

rand_m_num = 1

@app.post("/monitor_report")
async def monitor_report(request: Request) -> Response:
    request_dict = await request.json()
    # headers = request.headers
    #todo from header
    host = request_dict.pop("host")
    service_port = request_dict.pop("service_port")
    unfinished_reqs = request_dict.pop("unfinished_reqs")
    # unfinished_tokens = request_dict.pop("unfinished_tokens")
    used_gpu_blocks = request_dict.pop("used_gpu_blocks")
    used_cpu_blocks = request_dict.pop("used_cpu_blocks")
    remained_gpu_blocks = request_dict.pop("remained_gpu_blocks")
    remained_cpu_blocks = request_dict.pop("remained_cpu_blocks") 
    machine_type = request_dict.pop("machine_type")
    timestamp = request_dict.pop("timestamp")
    
    key = host + "_" + str(service_port) + "_" + machine_type
    # print(key, unfinished_req, unfinished_tokens)
    if instance_table.get(key):
        instance = instance_table[key]
        instance['unfinished_reqs'] = unfinished_reqs
        # instance['unfinished_tokens'] = unfinished_tokens
        instance['used_gpu_blocks'] = used_gpu_blocks
        instance['used_cpu_blocks'] = used_cpu_blocks
        instance['remained_gpu_blocks'] = remained_gpu_blocks
        instance['remained_cpu_blocks'] = remained_cpu_blocks
        instance['timestamp'] = timestamp
    else:
      instance = InstanceInfo(host, service_port, unfinished_reqs, 
                                used_gpu_blocks, used_cpu_blocks, remained_gpu_blocks, remained_cpu_blocks, machine_type, timestamp)
      instance_table[key] = instance.__dict__
    
    ##todo no instance response
    # if machine_type == MachineType.MPD:
    #     instances = {key: instance for key, instance in instance_table.items() if instance.machine_type == MachineType.MPD}
    # elif machine_type == MachineType.MPREFILL:
    #     instances = {key: instance for key, instance in instance_table.items() if instance.machine_type == MachineType.MDECODE}
    # elif machine_type  == MachineType.MDECODE:
    #     instances = {key: instance for key, instance in instance_table.items() if instance.machine_type == MachineType.MPREFILL}
        
    # return JSONResponse(instances)
    ret = {"result": 'monitor_report succ'}
    return ret

async def forward_request_to_server(request: Request, mprefill_forward_url, prefix_req=None) -> Response:
    # 获取原始请求的信息
    original_method = request.method
    original_headers = request.headers
    original_content = await request.body()
    if prefix_req:
        original_content_dict = json.loads(original_content)
        original_content_dict["prefix_req"] = prefix_req.__dict__
        original_content = json.dumps(original_content_dict)
    # 构建转发请求
    # forward_url = "http://example.com/destination_endpoint"
    async with httpx.AsyncClient() as client:
        forward_response = await client.request(
            original_method,
            mprefill_forward_url,
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
    forward_url = cfg.forward_res_url % (cfg.host_ip, cfg.client_port)
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
    request_dict = await request.json()
    request_id = request_dict.get("request_id")
    output_text = request_dict.get("output_text")
    mdecode_host = request_dict.get("mdecode_host")
    mdecode_port = request_dict.get("mdecode_port")

    req_cache = request_table[request_id]
    req_cache.mdecode_host = mdecode_host
    req_cache.mdecode_port = mdecode_port
    req_cache.add_output(output_text)
    for prompt, output in zip(req_cache.unprefilled_prompts, req_cache.outputs):
        req_cache.add_token(tokenizer(prompt).input_ids)
        req_cache.add_token(tokenizer(output).input_ids)
    while req_cache.unprefilled_prompts:
        req_cache.add_prefilled_prompt(req_cache.unprefilled_prompts.pop(0))
    trie.insert(req_cache.token, req_cache.request_id)
    #forward result to client
    response = await forward_result_to_client(request)
    ret = {"result": 'add_results succ'}
    return ret

@app.post("/add_request")
async def add_reqs(request: Request) -> Response:
    request_dict = await request.json()
    session_id = request_dict.get("session_id")
    request_id = request_dict.get("request_id")
    prompt = request_dict.get("prompt")
    prompt_token = tokenizer(prompt).input_ids
    matched_req_ids, matched_len = trie.search(prompt_token)
    #no matched other req
    if not matched_req_ids:
        req_cache = ReqCacheInfo(session_id=session_id, request_id=request_id)
        req_cache.add_unprefilled_prompt(prompt)
        mprefill_url, mprefill_host, mprefill_port  = random_choose_mp()# or load balance_choose_mp
        req_cache.mprefill_host = mprefill_host
        req_cache.mprefill_port = mprefill_port
        
        mapping = [mprefill_host, mprefill_port, None, None]
        req_machine_info[request_id] = mapping
        
        response = await forward_request_to_server(request, mprefill_url)
        request_table[request_id] = req_cache
        infight_req.add(request_id)
    else:
        mprefill_url, mprefill_host, mprefill_port, mdecode_host, mdecode_port,  prefix_request_id, matched_machine = matched_reqs_choose_mp(matched_req_ids)
        if matched_machine:
            prefix_req_cache = request_table[prefix_request_id]
            prefix_req = PrefixReqInfo(request_id=prefix_request_id, type=MachineType.MPREFILL, matched_len=matched_len,
                                    mdecode_host=prefix_req_cache.mdecode_host, mdecode_port=prefix_req_cache.mdecode_port, data_type=TransDataType.PART)
            reqs_prefix_table[request_id] = prefix_req
        else:
            prefix_req = PrefixReqInfo(request_id=prefix_request_id, type=MachineType.MDECODE, matched_len=matched_len,
                                mdecode_host=mdecode_host, mdecode_port=mdecode_port, data_type=TransDataType.FULL)
        
        mapping = [mprefill_host, mprefill_port, None, None]
        req_machine_info[request_id] = mapping
        
        req_cache = request_table[request_id]
        req_cache.mprefill_host = mprefill_host
        req_cache.mprefill_port = mprefill_port
        req_cache.add_token(prefix_req_cache.token[:matched_len])
        
        req_cache.add_unprefilled_prompt(prompt)
        response = await forward_request_to_server(request, mprefill_url, prefix_req)
        request_table[request_id] = req_cache
        infight_req.add(request_id)

    ret = {"text": 'add_request succ'}
    return ret

def matched_reqs_choose_mp(matched_req_ids):
    #todo 
    # choose one of matched_req_ids, if req's mprefill is overload, choose random
    req_id = matched_req_ids[0]
    mp_info = request_table[req_id]
    mprefill_url = cfg.forward_mprefill_url % (mp_info.mprefill_host, mp_info.mprefill_port)
    matched_machine = True
    return mprefill_url, mp_info.mprefill_host, mp_info.mprefill_host, mp_info.mdecode_host, mp_info.mdecode_host, req_id, matched_machine

def random_choose_mpd():
    host = ""
    service_port = ""
    mpd_instances = {key: instance for key, instance in instance_table.items() if instance.machine_type == MachineType.MPD}
    mpd = random.sample(mpd_instances.keys(), rand_m_num)
    mpd_instance = mpd_instances[mpd]
    host = mpd_instance.host
    service_port = mpd_instance.service_port
    url = cfg.forward_mpd_url % (host, service_port)
    return url, host, service_port

#todo 
def cache_aware_mpd():
    return 

def random_choose_mp():
    host = ""
    service_port = ""
    mp_instances = {key: instance for key, instance in instance_table.items() if instance.machine_type == MachineType.MPREFILL}
    mp = random.sample(mp_instances.keys(), rand_m_num)
    mp_instance = mp_instances[mp]
    host = mp_instance.host
    service_port = mp_instance.service_port
    url = cfg.forward_mprefill_url % (host, service_port)
    return url, host, service_port

def random_choose_md():
    host = ""
    service_port = ""
    md_instances = {key: instance for key, instance in instance_table.items() if instance.machine_type == MachineType.MDECODE}
    md = random.sample(md_instances.keys(), rand_m_num)
    md_instance = md_instances[md]
    host = md_instance.host
    service_port = md_instance.service_port
    url = cfg.forward_mdecode_url % (host, service_port)
    return url, host, service_port
#todo 
def cache_aware_mp():
    return

#todo 
def load_balacne_mp():
    return


#todo find md by load, transfer to add_kv_request
#todo by and matched len
@app.post("/add_kv_request")
async def add_kv_request(request: Request) -> Response:
    request_dict = await request.json()
    request_id = request_dict.get("request_id")
    md_url, md_host, md_port  = random_choose_md()
    mapping = req_machine_info[request_id]
    mapping[2] = md_host
    mapping[3] = md_port
    req_machine_info[request_id] = mapping
    response = await forward_request_to_server(request, md_url)
    
    ret = {"result": 'add_kv_request succ'}
    return ret

#todo need keep request id and m_p_d host mapping
#transfer to p response_kv_prepared
@app.post("/response_kv_prepared")
async def response_kv_prepared(request: Request) -> Response:
    request_dict = await request.json()
    request_id = request_dict.get("request_id")
    machine = req_machine_info[request_id]
    mp_host, mp_port = machine[0], machine[1]
    res_url = cfg.forward_mprefill_res_url % (mp_host, mp_port)
    response = await forward_request_to_server(request, res_url)
    
    ret = {"result": 'response_kv_prepared succ'}
    return ret

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
