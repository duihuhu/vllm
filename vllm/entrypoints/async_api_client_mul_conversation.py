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

async def async_post_http_request(
    prompt_token_ids: str,
    api_url: str,
    n: int = 1, 
    output_len: int = 16
):
    api_url = api_url
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
    }
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        print("post time ", time.time(), len(prompt_token_ids),  output_len)
        async with session.post(url=api_url, json=payload,
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
                        
async def post_request_and_get_response(args, prompts, interval):
    iteration = 0 
    history_value = []
    for prompt in prompts:
        if iteration == 0:
            time.sleep(interval)
        history_value.extend(prompt[0][0])
        output_len = prompt[0][1]
        response = async_post_http_request(history_value, G_URL, args.n, output_len)
        async for resp in response:
            resp = resp.decode('utf-8')
            resp = json.loads(resp)
            if resp['finished'] == True:
                history_value.extend(resp['prefilled_token_id'])
        iteration = iteration + 1 

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
