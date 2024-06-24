import json
from typing import Iterable, List, Optional, Tuple
import asyncio
import uuid
import aiohttp

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

    if args.test_type == "open":
        await asyncio.sleep(waiting_time)
    
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
    completion_token_ids = []
    async for resp in response:
        resp = resp.decode('utf-8')
        resp = json.loads(resp)
        if resp['n'] == 0:
            start_time = resp['start_time']
            ttft = resp['ttft']
            if resp['finished'] == True:
                end_time = resp['end_time']
                completion_token_ids.extend(resp['prefilled_token_id'])
        else:
            if resp['finished'] != True:
                tbt.append(resp['tbt'])
            elif resp['finished'] == True:
                end_time = resp['end_time']
                completion_token_ids.extend(resp['prefilled_token_id'])
    
    # print('completion_token_ids', completion_token_ids)
    # print('completion_token_ids length', len(completion_token_ids))
    # print('ground truth length', req[-1])
    assert len(completion_token_ids) == req[-1], 'Fail to keep the length of completion token ids'
        # yield (json.dumps(resp, ensure_ascii=False) + "\0").encode("utf-8")
    return (end_time-start_time, ttft, tbt[1:], tbt, tbt[0], req[-2] , req[-1], completion_token_ids)


async def dummy_post_request_and_get_response(args, req, waiting_time, **kwargs):

    if args.test_type == "open":
        await asyncio.sleep(waiting_time)
    
    main_request_id = kwargs['main_request_id']
    sub_request_id = kwargs['sub_request_id']
    print(f"main_request_id {main_request_id} sub_request_id {sub_request_id}")
    await asyncio.sleep(3) # Do some work here

    return (0.1, 1.1, [1.1, 2.2], [3.3, 4.4], 1.1, req[-2], req[-1], list(range(req[-1])))