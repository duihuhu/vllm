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
from vllm.transformers_utils.tokenizer import get_tokenizer
request_prompts_token_ids = {}
request_prompts = {}
G_URL = "http://127.0.0.1:8000/generate"  #GS服务器的地址 P


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


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

async def post_request_and_get_response(args, prompt, output_len):
    rsp = post_http_request(prompt, output_len, G_URL, args.n, args.stream)
    if args.stream:
        num_printed_lines = 0
        for h in get_streaming_response(rsp):
            print("res", h)
            # clear_line(num_printed_lines)
            # num_printed_lines = 0
            # for _, line in enumerate(h):
            #     num_printed_lines += 1
            #     print(f"vllm : {line!r}", flush=True)
                
async def main(args, prompts, output_lens):
    coroutines = []
    for prompt, output_len in zip(prompts, output_lens):
        # print(f"prompt:", end=' ', flush=True)
        # post_request_and_get_response(args, prompt)
        coroutines.append(asyncio.create_task(post_request_and_get_response(args, prompt, output_len)))
    await asyncio.gather(*coroutines)

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
    