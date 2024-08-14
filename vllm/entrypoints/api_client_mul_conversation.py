"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List, Optional, Tuple
import random
import requests
import asyncio
import time
import uuid
import numpy as np
import aiohttp
from transformers import PreTrainedTokenizerBase, AutoTokenizer

G_URL = "http://127.0.0.1:8081/add_request"  #GS服务器的地址 P

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


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
        session_len = 0
        while index < count:
            input_value =  conversations[index]['value']
            output_value =  conversations[index + 1]['value']
            input_value_token_ids = tokenizer(input_value).input_ids
            output_value_token_ids = tokenizer(output_value).input_ids
            # print("input_value_token_ids output_value_token_ids ", len(input_value_token_ids), len(output_value_token_ids))
            mul_turn.append((input_value_token_ids, len(output_value_token_ids)))
            session_info.append(mul_turn)
            session_len =  session_len + len(input_value_token_ids) +  len(output_value_token_ids)  
            index = index + 2
        if session_len < 4096:
            filtered_dataset.append(session_info)

    return filtered_dataset


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)



async def async_post_http_request(
    prompt_token_ids: str,
    api_url: str,
    n: int = 1, 
    output_len: int = 16
):
    api_url = api_url
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"User-Agent": "Test Client"}
        payload = {
            "prompt_token_ids": prompt_token_ids,
            "request_id": random_uuid(), 
            "n": n,
            "use_beam_search": False,
            "temperature": 0.0,
            "max_tokens": output_len,
            "logprobs": 1,
            "stream": True
            # "prompt_logprobs": 1
        }

        async with session.post(url=api_url, json=payload,
                                headers=headers) as response:
            if response.status == 200:
                print("response " , response.content)
                # async for chunk in response.content:
                #     chunk = chunk.strip()
                #     print("chunk " , response.content)
                #     if not chunk:
                #         continue
                async for chunk in response.iter_lines(chunk_size=8192,
                            decode_unicode=False,
                            delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode("utf-8"))
                        # output = data["text"]
                        print(data)
                        yield data


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
        "stream": True
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


async def post_request_and_get_response(args, prompts, interval):
    iteration = 0 
    history_value = []
    for prompt in prompts:
        if iteration == 0:
            print("post time ", time.time(), interval)
            time.sleep(interval)
        history_value.extend(prompt[0][0])
        output_len = prompt[0][1]
        # response = async_post_http_request(history_value, G_URL, args.n, output_len)
        iteration = iteration + 1
        rsp = post_http_request(history_value, G_URL, args.n, output_len)
        if args.stream:
            for h in get_streaming_response(rsp):
                if h['finished'] == True:
                    history_value.extend(h['prefilled_token_id'])
                    waiting_time = output_len * waiting_time_per_token / 1000
                    time.sleep(waiting_time)
    # return True    

async def main(args, prompts, reqs_interval):
    coroutines = []
    for prompt, interval in zip(prompts, reqs_interval):
        coroutines.append(asyncio.create_task(post_request_and_get_response(args, prompt, interval)))   
    await asyncio.gather(*coroutines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--session",  type=int, default=10)
    parser.add_argument("--request-rate",  type=int, default=10)

    args = parser.parse_args()
    tokenizer_path = "/home/jovyan/models/Llama-2-13b-hf/"

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    datasets = sample_requests("/home/jovyan/hucc/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 
                               tokenizer)

    reqs_interval = []
    pre_time = 0
    for i in range(args.session):
        interval = np.random.exponential(1.0 / args.request_rate)
        pre_time = pre_time + interval
        reqs_interval.append(pre_time)
        
    # print("reqs_interval ", reqs_interval)

    asyncio.run(main(args, datasets[:args.session], reqs_interval))
