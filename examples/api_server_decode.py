import argparse
import json
from typing import AsyncGenerator, List

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import time

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.outputs import RequestOutput
import threading
import socket
kv_data = {}
request_kv = {}

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
@app.post("/front_execute")
async def front_execute(background_tasks: BackgroundTasks) -> Response:
    ret = {"text": 'Start Decode'}
    background_tasks.add_task(background_execute)
    return JSONResponse(ret)

# @app.post("/background_execute")
def background_execute():
    print("start background execute ")
    global kv_data
    start_time_record = 0
    end_time_record = 0
    total_num_tokens = 0
    total_requests_compute = 128
    total_requests = total_requests_compute
    while True:
        outputs: List[RequestOutput] = []
        start_time = time.time()
        # while engine.engine.has_unfinished_requests():
        while engine.engine.has_unfinished_prefilled_seqs():
            print("background_execute kv data " ,len(kv_data), engine.engine.get_num_unfinished_requests())
            
            if start_time_record == 0:
                start_time_record = start_time
            step_outputs = engine.engine.step_decoder(kv_data, request_kv)
            for output in step_outputs:
                if output.finished:
                    print(output)
                    outputs.append(output)
        end_time = time.time()
        if len(outputs) !=0:
            total_requests = total_requests - len(outputs)
            total_num_tokens = total_num_tokens + sum(
                    len(output.outputs[0].token_ids)
                    for output in outputs
                )
            if total_requests == 0:
                end_time_record = end_time
                elapsed_time = end_time_record - start_time_record
                print("decode start time ", start_time_record)
                print("decode end time ", end_time_record)
                print("total_num_tokens ", total_num_tokens)
                print(end_time_record, start_time_record)
                print(f"Total {total_requests_compute} requests")
                print(f"Throughput: {total_requests_compute / elapsed_time:.2f} requests/s, "
                        f"{total_num_tokens / elapsed_time:.2f} tokens/s")
            #ret = {"text": 'Job Done'}
            #return JSONResponse(ret)
        #for output in outputs:
        #    prompt = output.prompt
        #    generated_text = output.outputs[0].text
        #    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

@app.post("/continuous_batching")
async def continous_batching(request: Request) -> Response:
    print(time.time(), "continuous_batching ")
    request_dict = await request.json()
    request_ids = request_dict.pop("request_ids")
    status = request_dict.pop("status")
    output_lens = request_dict.pop("output_lens")
    
    stream = request_dict.pop("stream", False)
    
    ret = {"text": 'Job Done'}
    # start_add_prefilled_request = time.time()
    # print("start_add_prefilled_request ", start_add_prefilled_request)
    if status == "prefilled":
        prompts = request_dict.pop("prompts")
        seq_ids = request_dict.pop("seq_ids")
        prompt_token_ids = request_dict.pop("prompt_token_ids")
        prefilled_token_ids = request_dict.pop("prefilled_token_ids")
        prefilled_texts = request_dict.pop("prefilled_texts")
        cumulative_logprobs = request_dict.pop("cumulative_logprobs")
        sampling_params_list = []
        for i in range(len(prompts)):
            sampling_params = SamplingParams(**request_dict)
            sampling_params_list.append(sampling_params)
        arrival_time = time.time()

        for prompt, prompt_token_id, request_id, seq_id, prefilled_token_id, prefilled_text, cumulative_logprob, output_len, sampling_param\
                in zip(prompts, prompt_token_ids, request_ids, seq_ids, prefilled_token_ids, prefilled_texts, cumulative_logprobs, output_lens, sampling_params_list):
            sampling_param.max_tokens = int(output_len)
            if engine.engine_use_ray:
                    engine.engine.add_prefilled_request.remote(
                        request_id,
                        prompt,
                        sampling_param,
                        seq_ids=seq_id,
                        prefilled_token_ids=prefilled_token_id,
                        prefilled_texts=prefilled_text,
                        cumulative_logprobs=cumulative_logprob,
                        prompt_token_ids=prompt_token_id,
                        arrival_time=arrival_time)
            else:
                    engine.engine.add_prefilled_request(
                        request_id,
                        prompt,
                        sampling_param,
                        seq_ids=seq_id,
                        prefilled_token_ids=prefilled_token_id,
                        prefilled_texts=prefilled_text,
                        cumulative_logprobs=cumulative_logprob,
                        prompt_token_ids=prompt_token_id,
                        arrival_time=arrival_time)
    
        # end_add_prefilled_request = time.time()
        # print("end_add_prefilled_request ", end_add_prefilled_request)

        return JSONResponse(ret)    
    else:
        return JSONResponse(ret)


@app.post("/mul_generate")
async def mul_generate(request: Request) -> Response:
    """Generate completion for the request, containing a list of prompts.

    The request should be a JSON object with the following fields:
    - prompts: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    request_ids = request_dict.pop("request_ids")
    output_lens = request_dict.pop("output_lens")
    status = request_dict.pop("status")
    stream = request_dict.pop("stream", False)
    print("status ", status)
    if status == 'start':
        prompts = request_dict.pop("prompts")
    elif status == 'prefilled':
        prompts = request_dict.pop("prompts")
        seq_ids = request_dict.pop("seq_ids")
        prompt_token_ids = request_dict.pop("prompt_token_ids")
        prefilled_token_ids = request_dict.pop("prefilled_token_ids")
        prefilled_texts = request_dict.pop("prefilled_texts")
        cumulative_logprobs = request_dict.pop("cumulative_logprobs")
    sampling_params_list = []
    for i in range(len(prompts)):
        sampling_params = SamplingParams(**request_dict)
        sampling_params_list.append(sampling_params)
    # sampling_params = SamplingParams(**request_dict)
    # # request_id = random_uuid()
    if status == 'start':
        results_generator = engine.mul_generate(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list,status=status)
    elif status == 'prefilled':
        results_generator = engine.mul_generate(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list,
                                                status=status, seq_ids=seq_ids, prompt_token_ids=prompt_token_ids, prefilled_token_ids=prefilled_token_ids,
                                                prefilled_texts=prefilled_texts, cumulative_logprobs=cumulative_logprobs)
        
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

def kv_server():
    global kv_data
    server_socket.listen(1)
    print("等待客户端连接...")
    while True:
        client_socket, client_address = server_socket.accept()
        # 设置发送缓冲区大小为 8192 字节
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192000)

        # 设置接收缓冲区大小为 8192 字节
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192000)
        print(f"连接来自: {client_address}", )
        
        req_len_bytes = client_socket.recv(8)
        req_id_length = int.from_bytes(req_len_bytes, byteorder='big')
        req_id_bytes = client_socket.recv(req_id_length)
        req_id = req_id_bytes.decode('utf-8')
        
        obj_count_bytes = client_socket.recv(8)
        obj_count = int.from_bytes(obj_count_bytes, byteorder='big')
        # 接收地址和长度
        while obj_count > 0:
            obj_len_bytes = client_socket.recv(8)
            obj_length = int.from_bytes(obj_len_bytes, byteorder='big')
            # Receive obj_bytes
            # print("obj_length ", obj_length)
            obj_id_bytes = client_socket.recv(obj_length)
            # print("obj_bytes ", type(obj_id_bytes), obj_id_bytes)
            obj = obj_id_bytes.decode('utf-8')
            # Receive kv_bytes
            kv_bytes_bytes = client_socket.recv(4)
            kv_bytes = int.from_bytes(kv_bytes_bytes, byteorder='big')
            data_bytes = client_socket.recv(kv_bytes)
            kv_data[obj] = data_bytes
            recv_buffer_size = client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            # print("decode obj ", obj, kv_bytes, "\n")
            # print("decode obj data ", recv_buffer_size, len(data_bytes), "\n")
            obj_count = obj_count - 1
        if req_id not in request_kv:
            request_kv[req_id] = 1
            print("kv_server ", time.time(), len(kv_data))
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args, "mdecode")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 12345)
    server_socket.bind(server_address)
    t_server = threading.Thread(target=kv_server)
    t_server.start()

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
