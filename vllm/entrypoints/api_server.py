"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import argparse
import json
import ssl
from typing import AsyncGenerator
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt_token_ids = request_dict.pop("prompt_token_ids")
    request_id = request_dict.pop("request_id")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    # print("max_tokens ", request_id, sampling_params.max_tokens)
    # request_id = random_uuid()
    start_time = time.time()
    results_generator = engine.generate(prompt=None,
                                        prompt_token_ids=prompt_token_ids, 
                                        sampling_params=sampling_params, 
                                        request_id=request_id)
    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        last_time = 0
        n = 0
        async for request_output in results_generator:
            if n == 0:
                end_time = time.time()
                ttft = end_time-start_time
                if request_output.finished != True:
                    ret = {"prefilled_token_id": request_output.outputs[0].token_ids, 
                           "finished": request_output.finished, "n": n, "ttft": ttft, "start_time": start_time, "end_time":end_time}
                else:
                    ret = {"prefilled_token_id": request_output.outputs[0].token_ids, 
                           "finished": request_output.finished, "n": n, "jct": ttft, "start_time": start_time, "end_time":end_time}
                    # print("jct ", request_id, n, len(request_output.outputs[0].token_ids), ttft)
            elif request_output.finished == True:
                end_time = time.time()
                jct = end_time-start_time
                ret = {"prefilled_token_id": request_output.outputs[0].token_ids, 
                    "finished": request_output.finished, "n": n, "jct": jct, "start_time": start_time, "end_time":end_time}
                # print("jct ",request_id, len(request_output.outputs[0].token_ids), jct)
            else:
                end_time = time.time()
                tbt = end_time - last_time
                last_time = end_time
                ret = {"prefilled_token_id": request_output.outputs[0].token_ids, 
                    "finished": request_output.finished, "n": n, "tbt": tbt, "start_time": start_time, "end_time": end_time}
            yield (json.dumps(ret) + "\0").encode("utf-8")
            n = n + 1

    if stream:
        return StreamingResponse(stream_results())

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
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
