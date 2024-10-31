"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List, Tuple
from transformers import PreTrainedTokenizerBase

import requests
import asyncio
import time
import uuid
import random
import aiohttp
from vllm.transformers_utils.tokenizer import get_tokenizer
request_prompts_token_ids = {}
request_prompts = {}
G_URL = "http://127.0.0.1:8000/generate"  #GS服务器的地址 P
AIOHTTP_TIMEOUT = 6 * 6 * 100

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)



async def asyc_forward_request(request_dict, api_url):
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
                        

def post_http_request(prompt: str,
                      output_len:int,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        # "request_id": random_uuid(), 
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": output_len,
        "logprobs": 1,
    }
    response = asyc_forward_request(pload, api_url)
        
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

async def post_request_and_get_response(prompt, output_len):
    pload = {
        "prompt": prompt,
        # "request_id": random_uuid(), 
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": output_len,
        "logprobs": 1,
    }
    response = asyc_forward_request(pload, G_URL)
    async for resp in response:
        resp = resp.decode('utf-8')
        resp = json.loads(resp)
        print("resp ", resp)
    return "1" 

async def main(args, prompts, output_lens):
    coroutines = []
    for prompt, output_len in zip(prompts, output_lens):
        coroutines.append(asyncio.create_task(post_request_and_get_response(prompt, output_len)))
    response = await asyncio.gather(*coroutines)

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[str]:
    random.seed(0)
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        # output_len = 16
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    # filtered_dataset: List[Tuple[str, int, int]] = []
    filtered_dataset: List[Tuple[str, List[int], str, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        request_id = random_uuid()
        prompt_len = len(prompt_token_ids)
        # if prompt_len > 256 or output_len > 128:
        if prompt_len < 4 or output_len < 4:
        # if prompt_len < 512 or output_len < 256:
        # if prompt_len > 512 or output_len < 128:
            # Prune too short sequences.
            continue
        # if prompt_len > 1024 or prompt_len + output_len > 2048:
        if prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, request_id, output_len))
        # filtered_prompts.append(prompt)
        # filtered_tokenids.append(prompt_token_ids)
    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)

    sampled_prompts = random.sample(filtered_dataset, num_requests)
    return sampled_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--num-servers", type=int, default=1)

      
    args = parser.parse_args()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = get_tokenizer(args.tokenizer)
    
    sampled_prompts = sample_requests(args.dataset, args.num_prompts, tokenizer)
    prompts = []
    request_ids = []
    output_lens = []
    for prompt in sampled_prompts:
      request_prompts[prompt[-2]] = prompt[0]
      request_prompts_token_ids[prompt[-2]] = prompt[-3]
      prompts.append(prompt[0])
      request_ids.append(prompt[-2])
      output_lens.append(prompt[-1])
      print(len(prompt[1]), prompt[-1])
    
    asyncio.run(main(args, prompts, output_lens))
    # prompts = ['San Francisco is a']
    # main(args,prompts)
    