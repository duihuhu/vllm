import argparse
import json

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import httpx
import random
import vllm.entrypoints.entrypoints_config as cfg
from typing import Dict, Set, List, Iterable, AsyncGenerator
import asyncio
import time
import requests


TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
tokenizer = None

async def forward_request_to_server(request_dict, api_url):
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=request_dict, stream=True)
    return response

def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            # output = data["text"]
            yield data

@app.post("/add_request")
async def add_request(request: Request) -> Response:
    print(" recv req time ", time.time())
    request_dict = await request.json()    
    #no matched other req
    generate_host, generate_port = cfg.generate_host, cfg.generate_port

    #提出 prefill repsonse内容text
    #forward_request_to_decode
    response = await forward_request_to_server(request_dict, cfg.forward_generate_url % 
                                                        (generate_host, generate_port))
    #return results to global scheduler
    async def stream_results() -> AsyncGenerator[bytes, None]:
        for res in get_streaming_response(response):
            yield (json.dumps(res, ensure_ascii=False) + "\0").encode("utf-8")
            
    return StreamingResponse(stream_results())

@app.post("/post")
async def test_time() -> Response:
    print(" recv req time ", time.time())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--workers", type=int, default=1)
    
    args = parser.parse_args()
    uvicorn.run(app='global_scheduler:app',
                host=cfg.global_scheduler_ip,
                port=cfg.global_scheduler_port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                workers=args.workers)
