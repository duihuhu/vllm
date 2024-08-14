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
import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
tokenizer = None

async def forward_request_to_server(request_dict, api_url):
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=request_dict, stream=True)
    return response

async def async_forward_request_to_server(request_dict, api_url):
    headers = {"User-Agent": "Test Client"}
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url=api_url, json=request_dict,
                                headers=headers) as response:
            if response.status == 200:
                delimiter=b"\0"
                buffer = b''  # 用于缓存数据块中的部分消息
                async for chunk in response.content.iter_any():
                    buffer += chunk  # 将新的数据块添加到缓冲区中
                    while delimiter in buffer:
                        index = buffer.index(delimiter)  # 查找分隔符在缓冲区中的位置
                        message = buffer[:index]  # 提取从缓冲区起始位置到分隔符位置的消息
                        yield message.strip()  # 返回提取的消息
                        buffer = buffer[index + len(delimiter):]  # 从缓冲区中移除已提取的消息和分隔符
                        
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
    response = async_forward_request_to_server(request_dict, cfg.forward_generate_url % 
                                                        (generate_host, generate_port))
    #return results to global scheduler
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for resp in response:
            resp = resp.decode('utf-8')
            resp = json.loads(resp)
            yield (json.dumps(resp, ensure_ascii=False) + "\0").encode("utf-8")
            
    return StreamingResponse(stream_results())

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
