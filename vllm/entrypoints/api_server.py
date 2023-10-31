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
    start_time_record = 0
    end_time_record = 0
    total_num_tokens = 0
    total_requests_compute = 2
    total_requests = total_requests_compute
    while True:
        outputs: List[RequestOutput] = []
        start_time = time.time()
        while engine.engine.has_unfinished_requests():
            if start_time_record == 0:
                start_time_record = start_time
            step_outputs = engine.engine.step_decoder()
            for output in step_outputs:
                if output.finished:
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
    request_dict = await request.json()
    request_ids = request_dict.pop("request_ids")
    status = request_dict.pop("status")
    output_lens = request_dict.pop("output_lens")
    
    stream = request_dict.pop("stream", False)
    
    ret = {"text": 'Job Done'}
    start_add_prefilled_request = time.time()
    print("start_add_prefilled_request ", start_add_prefilled_request)
    if status == "prefilled":
        prompts = request_dict.pop("prompts")
        seq_ids = request_dict.pop("seq_ids")
        prompt_token_ids = request_dict.pop("prompt_token_ids")
        prefilled_token_ids = request_dict.pop("prefilled_token_ids")
        prefilled_texts = request_dict.pop("prefilled_texts")
        cumulative_logprobs = request_dict.pop("cumulative_logprobs")
        sampling_params = SamplingParams(**request_dict)
        arrival_time = time.time()

        for prompt, prompt_token_id, request_id, seq_id, prefilled_token_id, prefilled_text, cumulative_logprob, output_len\
                in zip(prompts, prompt_token_ids, request_ids, seq_ids, prefilled_token_ids, prefilled_texts, cumulative_logprobs, output_lens):
            if engine.engine_use_ray:
                    sampling_params.max_tokens = int(output_len)
                    engine.engine.add_prefilled_request.remote(
                        request_id,
                        prompt,
                        sampling_params,
                        seq_ids=seq_id,
                        prefilled_token_ids=prefilled_token_id,
                        prefilled_texts=prefilled_text,
                        cumulative_logprobs=cumulative_logprob,
                        prompt_token_ids=prompt_token_id,
                        arrival_time=arrival_time)
            else:
                    sampling_params.max_tokens = int(output_len)
                    engine.engine.add_prefilled_request(
                        request_id,
                        prompt,
                        sampling_params,
                        seq_ids=seq_id,
                        prefilled_token_ids=prefilled_token_id,
                        prefilled_texts=prefilled_text,
                        cumulative_logprobs=cumulative_logprob,
                        prompt_token_ids=prompt_token_id,
                        arrival_time=arrival_time)
    
        end_add_prefilled_request = time.time()
        print("end_add_prefilled_request ", end_add_prefilled_request)

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

    sampling_params = SamplingParams(**request_dict)
    # # request_id = random_uuid()
    if status == 'start':
        results_generator = engine.mul_generate(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params,status=status)
    elif status == 'prefilled':
        results_generator = engine.mul_generate(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params,
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
