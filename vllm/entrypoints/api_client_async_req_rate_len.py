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
        # if prompt_len < 4 or output_len < 4:
        #     # Prune too short sequences.
        #     continue
        # if prompt_len < 2048 or prompt_len + output_len > 4096:
        #     # Prune too long sequences.
        #     continue
        if prompt_len < 2048 or output_len < 128 or output_len > 128 or prompt_len + output_len > 4096:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, prompt_len, output_len))

    print(len(filtered_dataset))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests



def random_uuid() -> str:
    return str(uuid.uuid4().hex)

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
                        
async def post_request_and_get_response(args, req, waiting_time):
    # print("waiting_time ", waiting_time)
    await asyncio.sleep(waiting_time)
    # print("post_request_and_get_response ", time.time())
    pload = {
        "prompt_token_ids": req[1],
        "request_id": random_uuid(), 
        "n": args.n,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": req[-1],
        "logprobs": 1,
        "ignore_eos": True,
        "stream":True
    }
    
    response = asyc_forward_request(pload, G_URL)
    start_time = 0
    end_time = 0
    ttft = 0
    tbt = []
    async for resp in response:
        resp = resp.decode('utf-8')
        resp = json.loads(resp)
        # print("resp ", resp)
        if resp['n'] == 0:
            start_time = resp['start_time']
            ttft = resp['ttft']
            if resp['finished'] == True:
                end_time = resp['end_time']
        else:
            if resp['finished'] != True:
                tbt.append(resp['tbt'])
            elif resp['finished'] == True:
                end_time = resp['end_time']
        # yield (json.dumps(resp, ensure_ascii=False) + "\0").encode("utf-8")
    return (end_time-start_time, ttft, tbt[1:], tbt[0], req[-2] , req[-1])

async def main(args, reqs):
    jct = []
    ttft = []
    tbt = []
    second_token = []
    waiting_time = 0
    coroutines = []
    for req in reqs:
        coroutines.append(asyncio.create_task(post_request_and_get_response(args, req, waiting_time)))
        interval = np.random.exponential(1.0 / args.request_rate)
        waiting_time = waiting_time + interval
    response = await asyncio.gather(*coroutines)
    for res in response:
        jct.append(res[0])
        ttft.append(res[1])
        tbt.extend(res[2])
        second_token.extend(res[3])
        # print("Res ", res)
    print("average jct , p90 jct,  p95 jct, average ttft , p90 ttft, p95 ttft, average tbt , p90 tbt, p95 tbt ", np.average(jct), np.percentile(jct, 90), np.percentile(jct, 95), np.average(ttft), np.percentile(ttft, 90), np.percentile(ttft, 95), np.average(tbt), np.percentile(tbt, 90), np.percentile(tbt, 95), np.average(second_token))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--request-rate", type=float, default=1)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=1)

    args = parser.parse_args()
    tokenizer_path = "/home/jovyan/models/Llama-2-13b-hf/"

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    random.seed(0)
    # reqs = sample_requests("/home/jovyan/hucc/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", tokenizer, args.num_requests)
    dummy_prompt_token_ids = np.random.randint(0, 10000, args.input_len)
    dummy_prompt_token_ids = dummy_prompt_token_ids.tolist()
    req = (0, dummy_prompt_token_ids, args.input_len, args.output_len)
    reqs = []
    for i in range(args.num_requests):
        reqs.append(req)

    # print("reqs ", reqs)    
    # reqs = [(None, [1,1,1,1,1], 5, 5), (None, [2,2,2,2], 4, 6)]
    asyncio.run(main(args, reqs))

    