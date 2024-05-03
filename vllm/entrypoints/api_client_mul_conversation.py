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

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    turn_conversations = 4
    dataset = [data for data in dataset if len(data["conversations"]) == turn_conversations]
    dataset = [data for data in dataset if len(data["conversations"]) % 2 == 0 ]
    dataset = [data for data in dataset if data["conversations"][0]["from"]=='human']

    filtered_dataset = []        
    # Filter out the conversations with less than 2 turns.
    for data in dataset:
        session_info = []
        mul_turn = []
        conversations = data["conversations"]
        count = len(conversations)
        index = 0 
        while index < count:
            input_value =  conversations[index]['value']
            output_value =  conversations[index + 1]['value']
            input_value_token_ids = tokenizer(input_value).input_ids
            output_value_token_ids = tokenizer(output_value).input_ids
            print("input_value_token_ids output_value_token_ids ", len(input_value_token_ids), len(output_value_token_ids))
            mul_turn.append((input_value_token_ids, len(output_value_token_ids)))
            session_info.append(mul_turn)
            index = index + 2
            
        filtered_dataset.append(session_info)

    return filtered_dataset


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
                      output_len: int = 16) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt_token_ids": prompt_token_ids,
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
    history_value = []
    for prompt in prompts:
        history_value.extend(prompt[0])
        output_len = prompt[1]
        rsp = post_http_request(history_value, G_URL, args.n, output_len)
        if args.stream:
            for h in get_streaming_response(rsp):
                if h['finished'] == True:
                    print("res ", h)
                    history_value.extend(h['prefilled_token_id'])
                    # waiting_time = output_len * waiting_time_per_token / 1000
                    # time.sleep(waiting_time)
                
def main(args, prompts):
    post_request_and_get_response(args, prompts)


async def main(args, prompts):
    coroutines = []
    for prompt in prompts:
        coroutines.append(asyncio.create_task(post_request_and_get_response(args, prompt)))
    await asyncio.gather(*coroutines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--session",  type=int, default=10)

    args = parser.parse_args()
    tokenizer_path = "/home/jovyan/models/Llama-2-13b-hf/"

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    datasets = sample_requests("/home/jovyan/hucc/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 
                               tokenizer)

    asyncio.run(main(args, datasets[:args.session]))
