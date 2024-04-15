"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List, Optional, Tuple
import random
import requests
import asyncio
import time
import uuid
from transformers import PreTrainedTokenizerBase, AutoTokenizer

G_URL = "http://127.0.0.1:8081/add_request"  #GS服务器的地址 P

#when repsone one token, waiting 100ms
waiting_time_per_token = 100
def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:

    filtered_dataset = []
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
        
    # Filter out the conversations with less than 2 turns.
    for data in dataset:
        conversations = data["conversations"]
    #     conver_tokens = 0
    #     for conver in conversations:
    #         value = conver['value']
    #         value_token_ids = tokenizer(value).input_ids
    #         conver_tokens  = conver_tokens + len(value_token_ids)
        count = len(conversations)
        index = 0 
        while index < count:
            input_value =  conversations[index]['value']
            output_value =  conversations[index + 1]['value']
            input_value_token_ids = tokenizer(input_value).input_ids
            output_value_token_ids = tokenizer(output_value).input_ids
            filtered_dataset.append((input_value, output_value, len(output_value_token_ids)))
            index = index + 2
    return filtered_dataset


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
                      output_len: int = 16) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "request_id": random_uuid(), 
        "n": n,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": output_len,
        "logprobs": 1,
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

def post_request_and_get_response(args, prompts):
    iteration = 0
    history_value = ""
    for prompt in prompts:
        input_prompt = history_value + prompt[0]
        output_len = prompt[2]
        rsp = post_http_request(input_prompt, G_URL, args.n, output_len)
        if args.stream:
            num_printed_lines = 0
            for h in get_streaming_response(rsp):
                # clear_line(num_printed_lines)
                # num_printed_lines = 0
                # for _, line in enumerate(h):
                #     num_printed_lines += 1
                #     print(f"vllm : {line!r}", flush=True)
                if h['finished'] == True:
                    print("res ", h)
                    history_value = history_value + prompt[0] + h['texts'][0]
                    waiting_time = output_len * waiting_time_per_token / 1000
                    time.sleep(waiting_time)
                
def main(args, prompts):
    post_request_and_get_response(args, prompts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    tokenizer_path = "/home/jovyan/models/Llama-2-13b-hf/"

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    datasets = sample_requests("/home/jovyan/vllm/vllm/entrypoints/one_conversation.json", tokenizer)
    

    main(args, datasets)

    