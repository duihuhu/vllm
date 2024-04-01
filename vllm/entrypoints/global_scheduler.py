import argparse
import json

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import httpx
import random
from vllm.entrypoints.global_trie_tree import Trie
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

async def forward_request_to_prefill(request_dict, api_url):
    headers = {"User-Agent": "Test Client"}
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

@app.post("/add_request")
async def add_request(request: Request) -> Response:
    request_dict = await request.json()
    # request_id = request_dict.get("request_id")
    # stream = request_dict.get("stream", True)
    # prompt = request_dict.get("prompt")
    # prompt_token = tokenizer(prompt).input_ids
    # matched_req_ids, matched_len = trie.search(prompt_token)
    matched_req_ids = None
    #no matched other req
    if not matched_req_ids:
        #提出 prefill repsonse内容text
        #forward_request_to_decode
        prefill_response = await forward_request_to_prefill(request_dict, cfg.forward_eprefill_url % (cfg.eprefill_host, cfg.eprefill_port))
        #提出 prefill repsonse内容text
        for res in get_streaming_response(prefill_response):
            prefill_res = res
            print("gs prefill_res ", prefill_res)
            
        #choose decode host and port(now is localhost), forward_request_to_decode generate_decode
        
        #decode_response = await forward_request_to_decode(prefill_res, cfg.forward_edecode_url % (cfg.edecode_host, cfg.edecode_port))
        decode_response = await forward_request_to_decode(prefill_res, cfg.forward_edecode_url % (cfg.edecode_host, \
            cfg.edecode_port if random.choice([True, False]) else cfg.edecode_port1))
        #decode_response

        print("stream_results stream_results ")
        #return results to global scheduler
        async def stream_results() -> AsyncGenerator[bytes, None]:
            # prefill' response, return to client
            n = 0
            yield (json.dumps(prefill_res) + "\0").encode("utf-8")

            for h in get_streaming_response(decode_response):
                print("res", h, n)
                #first send to prefll: add_response_kv_prepared
                if n == 0:
                    await send_to_prefill_response_kv_prepared(h, cfg.forward_eprefill_res_url % (cfg.eprefill_host, cfg.eprefill_port))
                else:
                    yield (json.dumps(h) + "\0").encode("utf-8")
                n = n + 1
        return StreamingResponse(stream_results())
        
        #prefill_response to send to d
        #generate decode
        
        #response to p
        
        #response token to client
        
        #update trie tree by token or by seqs
        
        request_table[request_id] = req_cache
        infight_req.add(request_id)
    


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
                host=cfg.global_scheduler_ip,
                port=cfg.global_scheduler_port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
