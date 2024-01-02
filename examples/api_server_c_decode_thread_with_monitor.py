import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import asyncio
import threading
import time
import mmap
import os
import requests
# import multiprocessing
# manager = multiprocessing.Manager()

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

mdecode_status = "init_mdecode_prefill"
 
decode_event = threading.Event()

dp_md = 'mdispatcher_to_mdecode.txt'

# if not os.path.isfile(dp_md):
    # create initial file
with open(dp_md, "w+b") as fd:
    fd.write(b'\x00' * 35 * 1024)

# init   
fd = open(dp_md, "r+b")
mm = mmap.mmap(fd.fileno(), 35 * 1024, access=mmap.ACCESS_WRITE, offset=0)

# @app.post("/notify_mdecode")
def notify_mdecode():
    global decode_event
    global mdecode_status
    hex_char = b'\x0F'
    # 判断内存
    prefill_nums = b'\x00'
    request_num =  b'\x00'
    already_num = 0 
    # 读取内存映射区域的数据
    while True:
        if prefill_nums != mm[(already_num*35):(already_num*35+1)]:
            prefill_nums = mm[(already_num*35):(already_num*35+1)]
            request_num = int.from_bytes(mm[(already_num*35+1):(already_num*35+2)], byteorder='big')
            request_id = mm[(already_num*35+2):(already_num*35+34)].decode("utf-8")
            label = int.from_bytes(mm[(already_num*35+34):(already_num*35+35)], byteorder='big')
            # print("mdecode recv signal from mprefill ", time.time())
            print("request info ", request_id, request_num, label)
            if request_num > 0:
                engine.convert_reqs_status_by_num(request_num)
                mdecode_status = "decode"
                already_num = already_num + 1
            decode_event.set()

def init_mdecode_prefill():
    global mdecode_status
    while True:
        # print("init_mdecode_prefill ", mdecode_status)
        if mdecode_status == "init_mdecode_prefill":
            decode_event.wait()
            results_generator = engine.generate_mdecode_prefill()
        elif mdecode_status == "decode":
            print("status is chanage, mdecode start exec decode", mdecode_status)
            engine.generate_decode()
        decode_event.clear()
        decode_event.wait()
        
@app.on_event("startup")
def startup_decode_event():
    threading.Thread(target=init_mdecode_prefill, daemon=True).start()
    threading.Thread(target=notify_mdecode, daemon=True).start()
    threading.Thread(target=monitor_mdecode_info, args=(args.host, args.port) ,daemon=True).start()

def post_monitor_request(monitor_url: str,
                      host: str,
                      service_port: int,
                      machine_type: str, 
                      num_labels: int ,
                      ) -> requests.Response:
    headers = {"User-Agent": "mdecode "}
    timestamp = time.time()
    pload = {
        "host": host,
        "service_port": service_port,
        "machine_type": machine_type,
        "num_labels": num_labels,
        "timestamp":timestamp,
    }
    response = requests.post(monitor_url, headers=headers, json=pload)
    
    return response

def post_mdecode_info(host, service_port, machine_type, num_labels):
    monitor_url = "http://127.0.0.1:9000/mdecode_monitor_report"
    response = post_monitor_request(monitor_url, host, service_port, machine_type, num_labels)

    
def monitor_mdecode_info(host, service_port):
    global engine
    machine_type = "decode"
    while True:
        num_labels = engine.monitor_mdecode_info()
        post_mdecode_info(host, service_port, machine_type, num_labels)
        time.sleep(0.1)


#background threads
@app.post("/init_mdecode")
async def init_mdecode(request: Request) -> Response:
    """init mdecode machine before execute. 
    add request to queue
    """
    request_dict = await request.json()
    
    request_ids = request_dict.pop("request_ids")
    prompts = request_dict.pop("prompts")
    output_lens = request_dict.pop("output_lens")
    stream = request_dict.pop("stream", False)
    sampling_params_list = []
    for i in range(len(prompts)):
        sampling_params = SamplingParams(**request_dict)
        sampling_params_list.append(sampling_params)
    engine.add_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
    decode_event.set()
    print("init_mdecode return ")
    ret = {"text": 'test'}
    return JSONResponse(ret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
