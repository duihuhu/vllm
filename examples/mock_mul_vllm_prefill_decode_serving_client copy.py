"""Example Python client for vllm.entrypoints.api_server"""
# python3 mul_vllm_serving_client.py --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/  --num-prompts 200 --num-servers 2
#python3 -m vllm.entrypoints.api_server --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/  --host 127.0.0.1 --port 8001 --tensor-parallel-size 2

import argparse
import json
from typing import Iterable, List, Tuple, Optional

from transformers import PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer import get_tokenizer
import requests
import random
import threading

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: List[str],
                      output_lens: List[int],
                      api_url: str,
                      n: int = 1,
                      stream: bool = False,
                      tid: Optional[int] = 0,
                      num_servers: Optional[int] = 1) -> requests.Response:
    if num_servers == 1:
        headers = {"User-Agent": "Test Client"}
        pload = {
            "prompt": prompt,
            "output_lens": output_lens,
            "n": 1,
            "use_beam_search": False,
            "temperature": 0.0,
            # "max_tokens": 16,
            'ignore_eos': True,
            "stream": stream,
        }
        response = requests.post(api_url, headers=headers, json=pload, stream=True)
    
    else:
        num_prompt = int(len(prompt)/num_servers)
        headers = {"User-Agent": "Test Client"}
        pload = {
            "prompt": prompt[tid*num_prompt:(tid+1)*num_prompt],
            "output_lens": output_lens[tid*num_prompt:(tid+1)*num_prompt],
            "n": 1,
            "use_beam_search": False,
            "temperature": 0.0,
            # "max_tokens": 16,
            'ignore_eos': True,
            "stream": stream,
        }
        response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output

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
    filtered_prompts: List[Tuple[str,int]] = [] 
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        # filtered_dataset.append((prompt, prompt_len, output_len))
        filtered_prompts.append((prompt, output_len))
    
    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)

    sampled_prompts = random.sample(filtered_prompts, num_requests)
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
    prompts = []
    output_lens = []
    sampled_prompts = sample_requests(args.dataset, args.num_prompts, tokenizer)
    for sampled_prompt in sampled_prompts:
        prompts.append(sampled_prompt[0])
        output_lens.append(sampled_prompt[-1]) 
    print(output_lens)
    # prompts = ["What is the easiest idea to earn money", "What is the easiest idea to earn money"]
    # prompt = args.prompt
    n = args.n
    stream = args.stream
    if args.num_servers == 1:
        api_url = f"http://{args.host}:{args.port}/mul_generate"
        response = post_http_request(prompts, output_lens, api_url, n, stream)
    else:
    # num_prompts = len(prompts)/(args.num_server)
    # print(f"Prompt: {prompts!r}\n", flush=True)
        threads = []
        for i in range(args.num_servers):
            api_url = f"http://{args.host}:{(args.port+i)}/mul_generate"
            threads.append(threading.Thread(target=post_http_request, args=(prompts, output_lens, api_url, n, stream, i, args.num_servers)))
        for td in threads:
            td.start()
        for td in threads:
            td.join()
    # if stream:
    #     num_printed_lines = 0
    #     for h in get_streaming_response(response):
    #         clear_line(num_printed_lines)
    #         num_printed_lines = 0
    #         for i, line in enumerate(h):
    #             num_printed_lines += 1
    #             print(f"Beam candidate {i}: {line!r}", flush=True)
    # else:
    #     output = get_response(response)
    #     for i, line in enumerate(output):
    #         print(f"Beam candidate {i}: {line!r}", flush=True)

