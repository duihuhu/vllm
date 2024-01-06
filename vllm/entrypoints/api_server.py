import argparse
import json
from typing import AsyncGenerator, List

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

@app.post("/add_reuqests_to_mul_generate_hhy")
async def add_reuqests_to_mul_generate_hhy(request: Request) -> Response:
    request_dict = await request.json()
    prompts_tokens_ids = request_dict.pop("prompts_tokens_ids")
    output_lens = request_dict.pop("output_lens")
    n = request_dict.pop("n")
    use_beam_search = request_dict.pop("use_beam_search")
    temperature = request_dict.pop("temperature")
    ignore_eos = request_dict.pop("ignore_eos")
    
    sampling_params: List[SamplingParams] = []
    for _ in range(len(output_lens)):
        sampling_param = SamplingParams(n = n,
                                        temperature = temperature,
                                        use_beam_search = use_beam_search,
                                        ignore_eos = ignore_eos)
        sampling_params.append(sampling_param)
    
    _ = engine.add_reuqests_to_mul_generate_hhy(output_lens = output_lens,
                                                sampling_params = sampling_params,
                                                prompts_tokens_ids = prompts_tokens_ids)
    
    ret = {"text": "test"}
    return JSONResponse(ret)

@app.post("/start_mul_generate_hhy")
async def start_mul_generate_hhy(backgroundtasks: BackgroundTasks):
    print("start mul_generate_hhy")
    backgroundtasks.add_task(do_mul_generate_hhy)
    return {"message": "start background tasks"}

async def do_mul_generate_hhy():
    while True:
        engine.mul_generate_hhy()


@app.post("/mul_generate")
async def mul_generate(request: Request) -> Response:
    """Generate completion for the request, containing a list of prompts.

    The request should be a JSON object with the following fields:
    - prompts: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompts = request_dict.pop("prompt")
    output_lens = request_dict.pop("output_lens")
    
    
    stream = request_dict.pop("stream", False)
    sampling_params_list = []
    for i in range(len(prompts)):
        sampling_params = SamplingParams(**request_dict)
        sampling_params_list.append(sampling_params)
    
    # # request_id = random_uuid()
    
    results_generator = engine.mul_generate(prompts, output_lens, sampling_params_list)

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
    parser.add_argument("--model", type=str, default="/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/")
    parser.add_argument("--tensor-parallel-size", type=int, default="2")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.model = args.model
    engine_args.tensor_parallel_size = args.tensor_parallel_size
    engine_args.max_num_batched_tokens = args.max_num_batched_tokens
    engine_args.max_num_seqs = args.max_num_seqs
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
