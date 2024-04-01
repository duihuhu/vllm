"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests
import asyncio
import time
import uuid

G_URL = "http://127.0.0.1:8081/add_request"  #GS服务器的地址 P


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "request_id": random_uuid(), 
        "n": n,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": 16,
        "logprobs": 1,
        "prompt_logprobs": 1
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            # output = data["text"]
            yield data


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output

async def post_request_and_get_response(args, prompt):
    rsp = post_http_request(prompt, G_URL, args.n, args.stream)
    if args.stream:
        num_printed_lines = 0
        for h in get_streaming_response(rsp):
            print("res", h)
            # clear_line(num_printed_lines)
            # num_printed_lines = 0
            # for _, line in enumerate(h):
            #     num_printed_lines += 1
            #     print(f"vllm : {line!r}", flush=True)
                
async def main(args, prompts):
    coroutines = []
    for prompt in prompts:
        print(f"prompt:", end=' ', flush=True)
        # post_request_and_get_response(args, prompt)
        coroutines.append(asyncio.create_task(post_request_and_get_response(args, prompt)))
    await asyncio.gather(*coroutines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompts = ['San Francisco is a', 'Where is Beijing?', 'Who is Bill Gates?']
    
    asyncio.run(main(args,prompts))
    # prompts = ['San Francisco is a']
    # main(args,prompts)
    