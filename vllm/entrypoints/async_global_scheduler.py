import argparse
import json

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import httpx
import random
from vllm.entrypoints.global_radix_tree import RadixCache
from vllm.entrypoints.global_meta import InstanceInfo, ReqCacheInfo, PrefixReqInfo, TransDataType
from vllm.entrypoints.comm import EngineType
from vllm.transformers_utils.tokenizer import get_tokenizer
import vllm.entrypoints.entrypoints_config as cfg
from typing import Dict, Set, List, Iterable, AsyncGenerator
import asyncio
import time
import requests


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
    global_ranks =  request_dict.pop("global_ranks") 
    timestamp = request_dict.pop("timestamp")
    key = host + "_" + str(port) + "_" + engine_type
    print("key global_ranks", key, global_ranks)
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
                              used_cpu_blocks, remained_gpu_blocks, remained_cpu_blocks, EngineType[engine_type], global_ranks, timestamp)
      instance_table[key] = instance

    ret = {"result": 'monitor_report succ'}
    return ret

async def forward_request_to_prefill(request_dict, api_url, cdecode_host=None, cdecode_port=None, cdecode_ranks=None, cdecode_blocks=None):
    headers = {"User-Agent": "Test Client"}
    if cdecode_host:
        request_dict['cmeta_host'] = cdecode_host
        request_dict['cmeta_port'] = cdecode_port
        request_dict['cmeta_ranks'] = cdecode_ranks
        request_dict['cmeta_kv_len'] = cdecode_blocks
        response = requests.post(api_url, headers=headers, json=request_dict, stream=True)
    else:
        response = requests.post(api_url, headers=headers, json=request_dict, stream=True)
    return response

async def forward_request_to_decode(prefill_res, api_url):
    headers = {"User-Agent": "Test Client"}
    pload = prefill_res
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response

async def send_to_prefill_response_kv_prepared(d_res, api_url):
    headers = {"User-Agent": "Test Client"}
    pload = d_res
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            # output = data["text"]
            yield data

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
        print("d_node ", d_node)
        print("d_tokens ", d_tokens, len(d_tokens), len(p_tokens))
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
    print("add request ", time.time())
    prompt_token_ids = request_dict["prompt_token_ids"]   
    #no matched other req
    eprefill_host, eprefill_port, cdecode_host, cdecode_port, cdecode_ranks,\
        edecode_host, edecode_port, cdecode_blocks = get_epd_cached_meta(gs_ptoken_tree, gs_dtoken_tree, prompt_token_ids)

    print("match prefill, decode, cdecode ", eprefill_host, edecode_host, cdecode_host)
    #提出 prefill repsonse内容text
    #forward_request_to_decode
    prefill_response = await forward_request_to_prefill(request_dict, cfg.forward_eprefill_url % 
                                                        (eprefill_host, eprefill_port), cdecode_host, cdecode_port, cdecode_ranks, cdecode_blocks)
    #提出 prefill repsonse内容text
    for res in get_streaming_response(prefill_response):
        prefill_res = res
        print("gs prefill_res ", prefill_res)
        
    #choose decode host and port(now is localhost), forward_request_to_decode generate_decode
    
    decode_response = await forward_request_to_decode(prefill_res, cfg.forward_edecode_url % 
                                                        (edecode_host, edecode_port))
    #decode_response
    print("stream_results stream_results ")
    #return results to global scheduler
    async def stream_results() -> AsyncGenerator[bytes, None]:
        # prefill' response, return to client
        n = 0
        prefilled_tokens = tuple(prefill_res["prompt_token_ids"] + prefill_res["prefilled_token_id"][:-1])
        gs_ptoken_tree.insert(prefilled_tokens, None, str(eprefill_host + "_" + str(eprefill_port)))
        
        yield (json.dumps(prefill_res, ensure_ascii=False) + "\0").encode("utf-8")

        for res in get_streaming_response(decode_response):
            #first send to prefll: add_response_kv_prepared
            if n == 0:
                await send_to_prefill_response_kv_prepared(res, cfg.forward_eprefill_res_url % 
                                                            (eprefill_host, eprefill_port))
            else:
                if res['finished'] == True:
                    decoded_tokens = tuple(res["prompt_token_ids"] + res["prefilled_token_id"][:-1])
                    gs_dtoken_tree.insert(decoded_tokens, None, str(edecode_host + "_" + str(edecode_port)))
                
                if res['finished'] == True and args.enable_dcache:
                    print("res", res, n)
                    decoded_tokens = tuple(res["prompt_token_ids"] + res["prefilled_token_id"])
                    gs_dtoken_tree.insert(decoded_tokens, None, str(edecode_host + "_" + str(edecode_port)))
                    pkv_response = await forward_request_to_prefill(res, cfg.forward_eprefill_res_kv_url % 
                                                                    (eprefill_host, eprefill_port))
                    for pkv_res in get_streaming_response(pkv_response):
                        await forward_request_to_decode(pkv_res, cfg.forward_edecode_res_kv_url  % 
                                                        (edecode_host, edecode_port))

                    #how to know data pass??
                    # gs_ptoken_tree.insert(decoded_tokens, None, str(cfg.edecode_host + ":" + cfg.edecode_port))

                yield (json.dumps(res, ensure_ascii=False) + "\0").encode("utf-8")
            n = n + 1
    return StreamingResponse(stream_results())


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="/workspace/opt-125m")
    parser.add_argument("--enable-dcache",  action="store_true", help=('enable pass decode to prefill cache '))
    
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = get_tokenizer(args.tokenizer)
    gs_ptoken_tree = RadixCache()
    gs_dtoken_tree = RadixCache()
    uvicorn.run(app,
                host=cfg.global_scheduler_ip,
                port=cfg.global_scheduler_port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
