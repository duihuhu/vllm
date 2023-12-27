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

mprefill_status_curr = "mprefill_execute"


 
mp_dp = 'mprefill_to_mdispatcher.txt'

if not os.path.isfile(mp_dp):
    # create initial file
    with open(mp_dp, "w+b") as fd:
        fd.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

mp_md_dp = 'mprefill_mdispatcher_to_mdecode_mdispatcher.txt'

if not os.path.isfile(mp_md_dp):
    # create initial file
    with open(mp_md_dp, "w+b") as fd:
        fd.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

prefill_event = threading.Event()

def mprefill_exec_prefill():
    with open(mp_dp, "r+b") as fd:
        mm = mmap.mmap(fd.fileno(), 8, access=mmap.ACCESS_WRITE, offset=0)
        prefill_nums = 0 
        while True:
            prefill_event.wait() 
            prefill_nums = engine.mprefill_generate_prefill(mm, prefill_nums)
            prefill_event.clear()
            prefill_event.wait()     
        mm.close()
        fd.close()
        
async def mprefill_add_prefill(request_dict):
    print("mprefill add prefill request ")
    request_ids = request_dict.pop("request_ids")
    prompts = request_dict.pop("prompts")
    output_lens = request_dict.pop("output_lens")
    stream = request_dict.pop("stream", False)
    mprefill_status = request_dict.pop("mprefill_status")
    sampling_params_list = []
    for i in range(len(prompts)):
        sampling_params = SamplingParams(**request_dict)
        sampling_params_list.append(sampling_params)
    # engine.add_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
    engine.add_mprefill_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
    prefill_event.set()

@app.post("/mprefill_add")
async def mprefill_add(request: Request) -> Response:
    request_dict = await request.json()
    await mprefill_add_prefill(request_dict)
    ret = {"text": 'test'}
    return JSONResponse(ret)

@app.on_event("startup")
def startup_decode_event():
    threading.Thread(target=mprefill_exec_prefill, daemon=True).start()

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
