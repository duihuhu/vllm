import argparse
import json

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import httpx
import api_global_scheduer_config as cfg
import random

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

#key: host_(service_port)_(machine_type)
#value: PrefillInfo object
monitor_mprefill_info = {}
monitor_mdecode_info = {}
monitor_mpd_info = {}

session_table = {}

rand_mpd = 1

#record session id , prompt, output, total text len, kv size, prefill machine, decode machine
class CacheInfo:
    def __init__(self, session_id) -> None:
        self.session_id = session_id
        self.src_url = None
        self.prompts = []
        self.outputs = []
        self.text_size = None
        self.kv_size = None
        self.mprefill_host = None
        self.mprefill_port = None
        self.mdecode_host = None
        self.mdecode_port = None
        self.pd_type = None
        pass

    def add_prompt(self, prompt) -> None:
        self.prompts.append(prompt)
        return  

    def add_output(self, output) -> None:
        self.outputs.append(output)
        return

class PDInfo:
    def __init__(self, host, service_port, unfinished_req, unfinished_tokens, timestamp) -> None:
        self.host = host
        self.service_port = service_port
        self.unfinished_req = unfinished_req
        self.unfinished_tokens = unfinished_tokens
        self.timestamp = timestamp
        
class PrefillInfo:
    def __init__(self, host, service_port, unfinished_req, unfinished_tokens, timestamp) -> None:
        self.host = host
        self.service_port = service_port
        self.unfinished_req = unfinished_req
        self.unfinished_tokens = unfinished_tokens
        self.timestamp = timestamp

class DecodeInfo:
    def __init__(self, host, service_port, machine_type, num_requests, timestamp) -> None:
        self.host = host
        self.service_port = service_port
        self.machine_type = machine_type
        self.num_requests = num_requests
        self.timestamp = timestamp
        
    def __json__(self):
        return {"host": self.host, "service_port": self.service_port, "machine_type": self.machine_type,
                "num_labels": self.num_labels, "timestamp": self.timestamp,}

@app.post("/mpd_monitor_report")
async def mpd_monitor_report(request: Request) -> Response:
    request_dict = await request.json()
    host = request_dict.pop("host")
    service_port = request_dict.pop("service_port")
    machine_type = request_dict.pop("machine_type")
    unfinished_req = request_dict.pop("unfinished_req")
    unfinished_tokens = request_dict.pop("unfinished_tokens")  
    timestamp = request_dict.pop("timestamp") 
    key = host + "_" + str(service_port) + "_" + machine_type
    ##todo
    if monitor_mpd_info.get(key):
        mpd_info = monitor_mpd_info[key]
        mpd_info.unfinished_req = unfinished_req
        mpd_info.unfinished_tokens = unfinished_tokens
        mpd_info.timestamp = timestamp
    else:
      mpd_info = PDInfo(host, service_port, unfinished_req, unfinished_tokens, timestamp)
      monitor_mpd_info[key] = mpd_info
    return JSONResponse(monitor_mpd_info)

@app.post("/mprefill_monitor_report")
async def mprefill_monitor_report(request: Request) -> Response:
    request_dict = await request.json()
    host = request_dict.pop("host")
    service_port = request_dict.pop("service_port")
    machine_type = request_dict.pop("machine_type")
    unfinished_req = request_dict.pop("unfinished_req")
    unfinished_tokens = request_dict.pop("unfinished_tokens")  
    timestamp = request_dict.pop("timestamp")    
    key = host + "_" + str(service_port) + "_" + machine_type
    if monitor_mprefill_info.get(key):
        mprefill_info = monitor_mprefill_info[key]
        mprefill_info.unfinished_req = unfinished_req
        mprefill_info.unfinished_tokens = unfinished_tokens
        mprefill_info.timestamp = timestamp
    else:
      mprefill_info = PrefillInfo(host, service_port, unfinished_req, unfinished_tokens, timestamp)
      monitor_mprefill_info[key] = mprefill_info
    return JSONResponse(monitor_mdecode_info)

@app.post("/mdecode_monitor_report")
async def mdecode_monitor_report(request: Request) -> Response:
    request_dict = await request.json()
    host = request_dict.pop("host")
    service_port = request_dict.pop("service_port")
    machine_type = request_dict.pop("machine_type")
    num_requests = request_dict.pop("num_requests") 
    timestamp = request_dict.pop("timestamp")    
    key = host + "_" + str(service_port) + "_" + machine_type
    # print(key, unfinished_req, unfinished_tokens)
    if monitor_mdecode_info.get(key):
        mdecode_info = monitor_mdecode_info[key]
        mdecode_info['machine_type'] = machine_type
        mdecode_info['num_requests'] = num_requests
        mdecode_info['timestamp'] = timestamp
    else:
      mdecode_info = DecodeInfo(host, service_port, machine_type, num_requests, timestamp)
      monitor_mdecode_info[key] = mdecode_info.__dict__
    ret = {"text": 'test'}
    return JSONResponse(ret)

def random_choose_mpd():
    host = ""
    service_port = ""
    mpd = random.sample(monitor_mpd_info.keys(), rand_mpd)
    pdinfo = monitor_mpd_info[mpd]
    host = pdinfo.host
    service_port = pdinfo.service_port
    url = cfg.forward_mpd_url % (host, service_port)
    return url, host, service_port
    
def choose_mprefill():
    host = ""
    service_port = ""
    # mprefill_local_add_request = "mprefill_add"
    min_unfinished_tokens = 0
    choice = None
    for key, value in monitor_mprefill_info.items():
        if choice == None:
            host = value.host
            service_port = value.service_port
            min_unfinished_tokens = value.unfinished_tokens
        else:
            if value.unfinished_tokens >=  min_unfinished_tokens:
                host = value.host
                service_port = value.service_port
    # mprefill_url =  "http://" + host + ":" + str(service_port) + "/" + mprefill_local_add_request
    url = cfg.forward_mprefill_url % (host, service_port)
    return url, host, service_port

async def forward_request_to_mprefill_server(request: Request, cache_info) -> Response:
    # 获取原始请求的信息
    original_method = request.method
    original_headers = request.headers
    original_content = await request.body()

    mprefill_forward_url, mprefill_host, mprefill_port  = choose_mprefill()
    cache_info.mprefill_host =  mprefill_host
    cache_info.mprefill_port = mprefill_port
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
    return response, cache_info

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

@app.post('/recv_decode_result')
async def recv_decode_result(request: Request) -> Response:
    request_dict = await request.json()
    session_id = request_dict.get("session_id")
    kv_size = request_dict.get("kv_size")
    output = request_dict.get("output")
    cache_info = session_table[session_id]
    cache_info.add_output(output)
    cache_info.kv_size = kv_size
    cache_info.text_size = cache_info.text_size + len(output)
    #forward result to client
    response = await forward_result_to_client(request)
    return 

@app.post("/add_reqs")
async def add_reqs(request: Request) -> Response:
    request_dict = await request.json()
    session_id = request_dict.get("session_id")
    prompt = request_dict.get("prompt")
    if session_id not in session_table:
        cache_info = CacheInfo(session_id=session_id)
        cache_info.add_prompt(prompt)
        cache_info.text_size = len(prompt)
        cache_info.kv_size = 0
        #first prompt from one session, choose mprefill and send
        # response, cache_info = await forward_request_to_mprefill_server(request, cache_info)
        session_table[session_id] = cache_info
        # return response
    else:
        cache_info = session_table[session_id]
        cache_info.add_prompt(prompt) 
        cache_info.text_size = cache_info.text_size + len(prompt)
        # response, cache_info = await forward_request_to_mprefill_server(request, cache_info)
        session_table[session_id] = cache_info
        ##after prompt, compose prompt and compute what to send
    ret = {"text": 'succ'}
    return ret



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    uvicorn.run(app,
                host=cfg.host_ip,
                port=cfg.global_scheduler_port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
