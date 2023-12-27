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
# import multiprocessing
# manager = multiprocessing.Manager()

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

mdecode_status = "init_mdecode_prefill"
mprefill_status_curr = "mprefill_execute"


 
mp_dp = 'mprefill_to_mdispatcher.txt'

if not os.path.isfile(mp_dp):
    # create initial file
    with open(mp_dp, "w+b") as fd:
        fd.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

# decode_event = asyncio.Event()

prefill_event = asyncio.Event()

decode_event = threading.Event()

@app.post("/notify_mdecode")
async def notify_mdecode(request: Request) -> Response:
    global decode_event
    global mdecode_status
    print("mdecode recv signal from mprefill ", time.time())
    request_dict = await request.json()
    request_ids = request_dict.pop("request_ids")
    engine.convert_reqs_status(request_ids)
    mdecode_status = "decode"
    decode_event.set()
    ret = {"text": 'test'}
    return JSONResponse(ret)

def init_mdecode_prefill(request_dict):
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
        
async def mprefill_exec_prefill(request_dict):
    global mprefill_status_curr
    with open(mp_dp, "r+b") as fd:
        mm = mmap.mmap(fd.fileno(), 8, access=mmap.ACCESS_WRITE, offset=0)
        prefill_nums = 0 
        while True:
            if mprefill_status_curr == "mprefill_execute":
                print("mprefill exec prefill request ")
                request_ids = request_dict.pop("request_ids")
                prompts = request_dict.pop("prompts")
                output_lens = request_dict.pop("output_lens")
                stream = request_dict.pop("stream", False)
                mprefill_status = request_dict.pop("mprefill_status")
                sampling_params_list = []
                for i in range(len(prompts)):
                    sampling_params = SamplingParams(**request_dict)
                    sampling_params_list.append(sampling_params)
                engine.add_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list, status=mprefill_status)
                results_generator = engine.mprefill_generate_prefill(mm, prefill_nums)
            elif mprefill_status_curr == "mprefill_add_exec":
                results_generator = engine.mprefill_generate_prefill(mm, prefill_nums)
            prefill_event.clear()
            await prefill_event.wait()     
        mm.close()
        fd.close()

async def mprefill_add_prefill(request_dict):
    print("mprefill add prefill request ")
    global mprefill_status_curr
    request_ids = request_dict.pop("request_ids")
    prompts = request_dict.pop("prompts")
    output_lens = request_dict.pop("output_lens")
    stream = request_dict.pop("stream", False)
    mprefill_status = request_dict.pop("mprefill_status")
    sampling_params_list = []
    for i in range(len(prompts)):
        sampling_params = SamplingParams(**request_dict)
        sampling_params_list.append(sampling_params)
    engine.add_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
    # results_generator = engine.generate_prefill(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list, status=mprefill_status)
    mprefill_status_curr = "mprefill_add_exec"
    prefill_event.set()

@app.post("/mprefill_add")
async def mprefill_add(request: Request) -> Response:
    request_dict = await request.json()
    await mprefill_add_prefill(request_dict)
    ret = {"text": 'test'}
    return JSONResponse(ret)
    
@app.post("/mprefill_execute")
async def mprefill_execute(request: Request) -> Response:
    request_dict = await request.json()
    background_task_future = asyncio.ensure_future(mprefill_exec_prefill(request_dict))
    ret = {"text": 'test'}
    return JSONResponse(ret)

#background threads
@app.post("/init_mdecode")
async def init_mdecode(request: Request, background_tasks: BackgroundTasks) -> Response:
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
