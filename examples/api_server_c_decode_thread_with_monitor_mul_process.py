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
import multiprocessing
# manager = multiprocessing.Manager()

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

mmap_event = multiprocessing.Event()

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

def get_request_from_mmap(request_queue):
    hex_char = b'\x0F'
    # 判断内存
    prefill_nums = b'\x00'
    request_num =  b'\x00'
    already_num = 0 
    # 读取内存映射区域的数据
    while True:
        if prefill_nums != mm[(already_num*35+34):(already_num*35+35)]:
            request_num = int.from_bytes(mm[(already_num*35):(already_num*35+1)], byteorder='big')
            request_id = mm[(already_num*35+1):(already_num*35+33)].decode("utf-8")
            label = int.from_bytes(mm[(already_num*35+33):(already_num*35+34)], byteorder='big')
            # arrive_time = time.time()
            request_queue.put([request_id, label])
            # time.sleep(0.000005)
            # add_time = time.time()
            # print("process decode get data " , request_id, arrive_time, add_time, add_time-arrive_time, "\n")
            already_num = already_num + 1

# def get_request_from_mmap_list(request_list):
#     hex_char = b'\x0F'
#     # 判断内存
#     prefill_nums = b'\x00'
#     request_num =  b'\x00'
#     already_num = 0 
#     # 读取内存映射区域的数据
#     while True:
#         if prefill_nums != mm[(already_num*35+34):(already_num*35+35)]:
#             request_num = int.from_bytes(mm[(already_num*35):(already_num*35+1)], byteorder='big')
#             request_id = mm[(already_num*35+1):(already_num*35+33)].decode("utf-8")
#             label = int.from_bytes(mm[(already_num*35+33):(already_num*35+34)], byteorder='big')
#             arrive_time = time.time()
#             request_list.append((request_id, label))
#             mmap_event.set()
#             add_time = time.time()
#             print("decode get data " , request_id, arrive_time, add_time, add_time-arrive_time)
#             already_num = already_num + 1
            
# # @app.post("/notify_mdecode")
# def notify_mdecode_from_list():
#     global decode_event
#     global mdecode_status
#     # 读取内存映射区域的数据

#     while True:
#         if len(request_list) == 0:
#             mmap_event.wait()
#         request_info = request_list.pop(0)
#         arrive_time = time.time()
#         print("decode get data " , request_info[0], arrive_time)
#         engine.convert_req_label_status(request_info[0], request_info[1])
#         mdecode_status = "decode"
#         decode_event.set()
        


# @app.post("/notify_mdecode")
def notify_mdecode_from_queue():
    global decode_event
    global mdecode_status
    # 读取内存映射区域的数据

    while True:
        request_info = request_queue.get()
        arrive_time = time.time()
        # print("queue decode get data " , request_info[0], arrive_time, "\n")
        # engine.convert_req_label_status(request_info[0], request_info[1], arrive_time=arrive_time-0.0015)
        engine.convert_req_label_status(request_info[0], request_info[1], arrive_time=arrive_time-0.0015)
        mdecode_status = "decode"
        decode_event.set()
        
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
        if prefill_nums != mm[(already_num*35+34):(already_num*35+35)]:
            # prefill_nums = mm[(already_num*35):(already_num*35+1)]
            # start = time.time()
            request_num = int.from_bytes(mm[(already_num*35):(already_num*35+1)], byteorder='big')
            request_id = mm[(already_num*35+1):(already_num*35+33)].decode("utf-8")
            label = int.from_bytes(mm[(already_num*35+33):(already_num*35+34)], byteorder='big')
            arrive_time = time.time()
            # print("decode get data " , request_id, start, end, end-start)
            # print("mdecode recv signal from mprefill ", time.time())
            # print("request info ", request_id, request_num, label, time.time())
            # if request_num > 0:
            # engine.convert_reqs_status_by_num(request_num)
            # engine.convert_reqs_status(request_id)
            engine.convert_req_label_status(request_id, label, arrive_time)
            # add_time = time.time()
            # print("decode get data " , request_id, arrive_time, add_time, add_time-arrive_time)

            mdecode_status = "decode"
            already_num = already_num + 1
            decode_event.set()
            if already_num >= 128:
                time.sleep(500)
    

def init_mdecode_prefill():
    global mdecode_status
    total_time = 0
    accomplish_request_num = 0
    while True:
        # print("init_mdecode_prefill ", mdecode_status)
        if mdecode_status == "init_mdecode_prefill":
            decode_event.wait()
            results_generator = engine.generate_mdecode_prefill()
        elif mdecode_status == "decode":
            print("status is chanage, mdecode start exec decode", mdecode_status)
            start_time = time.time()
            acc_complish = engine.generate_decode()
            end_time = time.time()
            accomplish_request_num = accomplish_request_num + acc_complish 
            # print("decode time ", start_time, end_time ,start_time-end_time)
            total_time = total_time + end_time-start_time
        if accomplish_request_num == 128:
            print("machine decode accomplish time ", time.time(), total_time)
        decode_event.clear()
        decode_event.wait()
        
@app.on_event("startup")
def startup_decode_event():
    threading.Thread(target=init_mdecode_prefill, daemon=True).start()
    # threading.Thread(target=notify_mdecode, daemon=True).start()
    threading.Thread(target=notify_mdecode_from_queue, daemon=True).start()
    # threading.Thread(target=notify_mdecode_from_list, daemon=True).start()
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
        time.sleep(1000)


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
   
    request_queue = multiprocessing.Queue()
    request_queue.put([0,0])
    request_queue.get()
    mmap_process = multiprocessing.Process(target=get_request_from_mmap, args=(request_queue,))
    mmap_process.start()
    
    # request_list = multiprocessing.Manager().list()
    # mmap_process = multiprocessing.Process(target=get_request_from_mmap_list, args=(request_list,))
    # mmap_process.start()
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
