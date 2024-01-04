import argparse
import json
import multiprocessing
import mmap
import os
import threading
import time
from typing import AsyncGenerator

import requests
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()

mdecode_status = "init_mdecode_prefill"
decode_event = multiprocessing.Event()
dp_md = 'mdispatcher_to_mdecode.txt'

if not os.path.isfile(dp_md):
    with open(dp_md, "w+b") as fd:
        fd.write(b'\x00' * 35 * 1024)

fd = open(dp_md, "r+b")
mm = mmap.mmap(fd.fileno(), 35 * 1024, access=mmap.ACCESS_WRITE, offset=0)

# 使用 multiprocessing.Value 共享 engine
shared_engine = None

def init_global_engine(args, shared_engine):
    global engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    shared_engine.value = engine  # 将 engine 赋值给共享对象

def notify_mdecode():
    global decode_event
    global mdecode_status
    hex_char = b'\x0F'
    prefill_nums = b'\x00'
    request_num = b'\x00'
    already_num = 0

    while True:
        if prefill_nums != mm[(already_num*35+34):(already_num*35+35)]:
            request_num = int.from_bytes(mm[(already_num*35):(already_num*35+1)], byteorder='big')
            request_id = mm[(already_num*35+1):(already_num*35+33)].decode("utf-8")
            label = int.from_bytes(mm[(already_num*35+33):(already_num*35+34)], byteorder='big')
            arrive_time = time.time()

            engine.convert_req_label_status(request_id, label)
            add_time = time.time()
            print("decode get data ", request_id, arrive_time, add_time, add_time-arrive_time)

            mdecode_status = "decode"
            already_num = already_num + 1
            decode_event.set()

def init_mdecode_prefill():
    global mdecode_status
    while True:
        if mdecode_status == "init_mdecode_prefill":
            decode_event.wait()
            results_generator = shared_engine.value.generate_mdecode_prefill()  # 使用共享的 engine
        elif mdecode_status == "decode":
            print("status is chanage, mdecode start exec decode", mdecode_status)
            shared_engine.value.generate_decode()  # 使用共享的 engine
        decode_event.clear()
        decode_event.wait()

def post_monitor_request(monitor_url: str,
                         host: str,
                         service_port: int,
                         machine_type: str,
                         num_labels: int) -> requests.Response:
    headers = {"User-Agent": "mdecode "}
    timestamp = time.time()
    pload = {
        "host": host,
        "service_port": service_port,
        "machine_type": machine_type,
        "num_labels": num_labels,
        "timestamp": timestamp,
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
        num_labels = shared_engine.value.monitor_mdecode_info()  # 使用共享的 engine
        post_mdecode_info(host, service_port, machine_type, num_labels)
        time.sleep(1000)

# background processes
def init_mdecode_process():
    init_mdecode_prefill()

def notify_mdecode_process():
    notify_mdecode()

def monitor_mdecode_info_process(host, service_port):
    monitor_mdecode_info(host, service_port)

@app.on_event("startup")
def startup_decode_event():
    global shared_engine
    shared_engine = multiprocessing.Value('P', None)  # 使用 multiprocessing.Value 共享 engine
    with multiprocessing.Manager() as manager:
        init_global_engine(args, shared_engine)

        multiprocessing.Process(target=init_mdecode_process, daemon=True).start()
        multiprocessing.Process(target=notify_mdecode_process, daemon=True).start()
        multiprocessing.Process(target=monitor_mdecode_info_process, args=(args.host, args.port), daemon=True).start()

# background processes endpoint
@app.post("/init_mdecode_process")
async def init_mdecode_process_endpoint(request: Request) -> Response:
    multiprocessing.Process(target=init_mdecode_process, daemon=True).start()
    ret = {"text": 'init_mdecode_process started'}
    return JSONResponse(ret)

@app.post("/notify_mdecode_process")
async def notify_mdecode_process_endpoint(request: Request) -> Response:
    multiprocessing.Process(target=notify_mdecode_process, daemon=True).start()
    ret = {"text": 'notify_mdecode_process started'}
    return JSONResponse(ret)

@app.post("/monitor_mdecode_info_process")
async def monitor_mdecode_info_process_endpoint(request: Request) -> Response:
    multiprocessing.Process(target=monitor_mdecode_info_process, args=(args.host, args.port), daemon=True).start()
    ret = {"text": 'monitor_mdecode_info_process started'}
    return JSONResponse(ret)

# your existing route
@app.post("/init_mdecode")
async def init_mdecode(request: Request) -> Response:
    request_dict = await request.json()

    request_ids = request_dict.pop("request_ids")
    prompts = request_dict.pop("prompts")
    output_lens = request_dict.pop("output_lens")
    stream = request_dict.pop("stream", False)
    sampling_params_list = []
    for i in range(len(prompts)):
        sampling_params = SamplingParams(**request_dict)
        sampling_params_list.append(sampling_params)
    shared_engine.value.add_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)  # 使用共享的 engine
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

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
