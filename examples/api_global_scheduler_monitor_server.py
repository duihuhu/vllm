import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import httpx

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

#key: host_(service_port)_(machine_type)
#value: PrefillInfo object
monitor_mprefill_info = {}

monitor_mdecode_info = {}

class PrefillInfo:
    def __init__(self, host, service_port, unfinished_req, unfinished_tokens, timestamp) -> None:
        self.host = host
        self.service_port = service_port
        self.unfinished_req = unfinished_req
        self.unfinished_tokens = unfinished_tokens
        self.timestamp = timestamp

class DecodeInfo:
    def __init__(self, host, service_port, num_labels, timestamp) -> None:
        self.host = host
        self.service_port = service_port
        self.num_labels = num_labels
        self.timestamp = timestamp

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
    print(key, unfinished_req, unfinished_tokens)
    if monitor_mprefill_info.get(key):
        mprefill_info = monitor_mprefill_info[key]
        mprefill_info.unfinished_req = unfinished_req
        mprefill_info.unfinished_tokens = unfinished_tokens
        mprefill_info.timestamp = timestamp
    else:
      mprefill_info = PrefillInfo(host, service_port, unfinished_req, unfinished_tokens, timestamp)
      monitor_mprefill_info[key] = mprefill_info
    # ret = {"mdecode_info": monitor_mdecode_info}
    return JSONResponse(content=monitor_mdecode_info)

@app.post("/mdecode_monitor_report")
async def mdecode_monitor_report(request: Request) -> Response:
    request_dict = await request.json()
    host = request_dict.pop("host")
    service_port = request_dict.pop("service_port")
    machine_type = request_dict.pop("machine_type")
    num_labels = request_dict.pop("num_labels") 
    timestamp = request_dict.pop("timestamp")    
    key = host + "_" + str(service_port) + "_" + machine_type
    # print(key, unfinished_req, unfinished_tokens)
    if monitor_mdecode_info.get(key):
        mdecode_info = monitor_mdecode_info[key]
        mdecode_info.num_labels = num_labels
        mdecode_info.timestamp = timestamp
    else:
      mdecode_info = DecodeInfo(host, service_port, num_labels, timestamp)
      monitor_mdecode_info[key] = mdecode_info
    ret = {"text": 'test'}
    return JSONResponse(ret)

def compose_mprefill_url():
    host = ""
    service_port = ""
    mprefill_local_add_request = "mprefill_add"
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
    mprefill_url =  "http://" + host + ":" + str(service_port) + "/" + mprefill_local_add_request
    return mprefill_url

async def forward_request_to_another_server(request: Request) -> Response:
    # 获取原始请求的信息
    original_method = request.method
    original_url = request.url
    original_headers = request.headers
    original_content = await request.body()

    forward_url = compose_mprefill_url()
    print(forward_url)
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

@app.post("/global_prefill_req_pool")
async def global_prefill_req_pool(request: Request) -> Response:
    # 转发请求到另一个服务器
    response = await forward_request_to_another_server(request)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
