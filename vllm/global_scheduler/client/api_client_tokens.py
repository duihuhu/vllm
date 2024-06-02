"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests
import asyncio
import time
import uuid
from transformers import PreTrainedTokenizerBase, AutoTokenizer

G_URL = "http://127.0.0.1:8081/add_request"  #GS服务器的地址 P


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt_token_ids: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt_token_ids": prompt_token_ids,
        "request_id": random_uuid(), 
        "n": n,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": 16,
        "logprobs": 1,
        "stream":True
        # "prompt_logprobs": 1
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

def post_request_and_get_response(args, prompt):
    rsp = post_http_request(prompt, G_URL, args.n, args.stream)
    if args.stream:
        num_printed_lines = 0
        for h in get_streaming_response(rsp):
            if h['finished'] == True:
                print("res", h)
                return h["prefilled_token_id"]
            # clear_line(num_printed_lines)
            # num_printed_lines = 0
            # for _, line in enumerate(h):
            #     num_printed_lines += 1
            #     print(f"vllm : {line!r}", flush=True)
            
                
def main(args, prompts):
    # coroutines = []
    # for prompt in prompts:
    #     print(f"prompt:", end=' ', flush=True)
    prefilled_token_id = post_request_and_get_response(args, prompts)
    # prefilled_token_id = post_request_and_get_response(args, prompts + prefilled_token_id)

    #     coroutines.append(asyncio.create_task(post_request_and_get_response(args, prompt)))
    # await asyncio.gather(*coroutines)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--seq-lens", type=int, default=1)
    
    # prompts = ['San Francisco is a', 'Where is Beijing?', 'Who is Bill Gates?']
    
    tokenizer_path = "/home/jovyan/models/Llama-2-13b-hf/"
    args = parser.parse_args()

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # asyncio.run(main(args,prompts))
    # prompts = ['San Francisco is a San Francisco is a San Francisco is a']
    warm_prompts = ['111111']
    warm_value_token_ids = tokenizer(warm_prompts[0]).input_ids[1:]
    for i in range(20):
        main(args, warm_value_token_ids)
    
    input_prompt = 'San Francisco'

    for i in range(args.seq_lens):
        input_prompt = input_prompt + " " + 'San Francisco'
    prompts = [input_prompt]
    input_value_token_ids = tokenizer(prompts[0]).input_ids[1:]

    main(args,input_value_token_ids)
    