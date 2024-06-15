import argparse
import json
from typing import Iterable, List, Optional, Tuple
import asyncio
import uuid
import aiohttp
import numpy as np 
import torch
import random

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
G_URL = "http://127.0.0.1:8081/add_request"  #GS服务器的地址 P

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
    return (end_time-start_time, ttft, tbt[1:], tbt, tbt[0], req[-2] , req[-1])

async def run(args, reqs):
    jct = []
    ttft = []
    tbt_no_second_token = []
    tbt_with_second_token = []
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
        tbt_no_second_token.extend(res[2])
        tbt_with_second_token.extend(res[3])
        second_token.append(res[4])
        # print("Res ", res)
    print("average_jct, p99_jct, average_ttft, p99_ttft, average_tbt_no_second_token, p99_tbt_no_second_token, average_tbt_with_second_token, p99_tbt_with_second_token")
    print("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(np.average(jct), np.percentile(jct, 99), np.average(ttft), np.percentile(ttft, 99), np.average(tbt_no_second_token), np.percentile(tbt_no_second_token, 99), np.average(tbt_with_second_token), np.percentile(tbt_with_second_token, 99)))

def get_args() -> argparse.Namespace:
    
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
    parser.add_argument("--dataset", type=str, default="ShareGPT", choices=["ShareGPT", "LooGLE", "ReAct"])

    args = parser.parse_args()

    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False