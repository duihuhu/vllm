"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List, Optional, Tuple
import random
import requests
import asyncio
import time
import uuid
import aiohttp
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import numpy as np 
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

G_URL = "http://127.0.0.1:8081/add_request"  #GS服务器的地址 P

#when repsone one token, waiting 100ms
waiting_time_per_token = 100
def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int 
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 4096:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests



def random_uuid() -> str:
    return str(uuid.uuid4().hex)

async def asyc_forward_request_resp(request_dict, api_url):
    headers = {"User-Agent": "Test Client"}
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url=api_url, json=request_dict,
                                headers=headers) as response:
            return await response.text()

async def post_request_and_get_response(args, req):
    pload = {
        "prompt_token_ids": req[1],
        "request_id": random_uuid(), 
        "n": args.n,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": req[-1],
        "logprobs": 1,
        "stream":True
    }
    resp = await asyc_forward_request_resp(pload, G_URL)
    print("resp ", resp)
    return resp
async def main(args, reqs):
    waiting_time = 0
    coroutines = []
    for req in reqs:
        coroutines.append(asyncio.create_task(post_request_and_get_response(args, req)))
        interval = np.random.exponential(1.0 / args.request_rate)
        waiting_time = waiting_time + interval
        time.sleep(waiting_time)
    await asyncio.gather(*coroutines)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--request-rate", type=float, default=0.1)
    parser.add_argument("--num-requests", type=int, default=1)
    args = parser.parse_args()
    tokenizer_path = "/home/jovyan/models/Llama-2-13b-hf/"

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # reqs = sample_requests("/home/jovyan/hucc/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", tokenizer, args.num_requests)
    
    reqs = [(None, [1,1,1,1], 4, 6), (None, [2,2,2,2], 4, 6)]
    asyncio.run(main(args, reqs))

    