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
# import multiprocessing
# manager = multiprocessing.Manager()

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

mdecode_status = "init_mdecode_prefill"
mprefill_status_curr = "mprefill_execute"

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
            # request_ids = request_dict.pop("request_ids")
            # prompts = request_dict.pop("prompts")
            # output_lens = request_dict.pop("output_lens")
            # stream = request_dict.pop("stream", False)
            # sampling_params_list = []
            # for i in range(len(prompts)):
            #     sampling_params = SamplingParams(**request_dict)
            #     sampling_params_list.append(sampling_params)
            results_generator = engine.generate_mdecode_prefill()
        elif mdecode_status == "decode":
            print("status is chanage, mdecode start exec decode", mdecode_status)
            engine.generate_decode()
        decode_event.clear()
        decode_event.wait()
        
# @app.on_event("startup")
# def startup_decode_event():
#     threading.Thread(target=init_mdecode_prefill, daemon=True).start()
        
async def mprefill_exec_prefill(request_dict):
    global mprefill_status_curr
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
            results_generator = engine.generate_prefill(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list, status=mprefill_status)
        elif mprefill_status_curr == "mprefill_add":
            results_generator = engine.generate_prefill(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list, status=mprefill_status)
        prefill_event.clear()
        await prefill_event.wait()

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
    results_generator = engine.generate_prefill(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list, status=mprefill_status)
    mprefill_status_curr = "mprefill_add"
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
    """Generate completion for the request, containing a list of prompts.

    The request should be a JSON object with the following fields:
    - prompts: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
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
    results_generator = engine.generate_prefill(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list, status="")
    decode_event.set()
    
    # background_task_future = asyncio.ensure_future(init_mdecode_prefill(request_dict))
    
    # background_tasks.add_task(init_mdecode_prefill(request_dict))
    # thread = threading.Thread(target=init_mdecode_prefill, args=(request_dict))
    # thread.start()
    # prompts = request_dict.pop("prompt")
    # output_lens = request_dict.pop("output_lens")
    
    
    # stream = request_dict.pop("stream", False)
    # sampling_params_list = []
    # for i in range(len(prompts)):
    #     sampling_params = SamplingParams(**request_dict)
    #     sampling_params_list.append(sampling_params)
    
    # # # request_id = random_uuid()
    
    # results_generator = engine.mul_generate(prompts, output_lens, sampling_params_list)

    # # Streaming case
    # async def stream_results() -> AsyncGenerator[bytes, None]:
    #     async for request_output in results_generator:
    #         prompt = request_output.prompt
    #         text_outputs = [
    #             prompt + output.text for output in request_output.outputs
    #         ]
    #         ret = {"text": text_outputs}
    #         yield (json.dumps(ret) + "\0").encode("utf-8")

    # async def abort_request() -> None:
    #     await engine.abort(request_id)

    # if stream:
    #     background_tasks = BackgroundTasks()
    #     # Abort the request if the client disconnects.
    #     background_tasks.add_task(abort_request)
    #     return StreamingResponse(stream_results(), background=background_tasks)

    # # Non-streaming case
    # final_output = None
    # async for request_output in results_generator:
    #     if await request.is_disconnected():
    #         # Abort the request if the client disconnects.
    #         await engine.abort(request_id)
    #         return Response(status_code=499)
    #     final_output = request_output

    # assert final_output is not None
    # prompt = final_output.prompt
    # text_outputs = [prompt + output.text for output in final_output.outputs]
    # ret = {"text": text_outputs}
    print("init_mdecode return ")
    ret = {"text": 'test'}
    return JSONResponse(ret)

@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
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
