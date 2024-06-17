from typing import Iterable, List, Optional, Tuple
import asyncio
import numpy as np 
import time
from .common import post_request_and_get_response, dummy_post_request_and_get_response

'''
Warning: The code snippet in this file is purely asynchronous and single-threaded.
And that is why we are going wild in using global variables.
Please do not follow the code convention in this file in other settings.
'''
waiting_time = 0
time_start = time.perf_counter()
response = []

# Handle all subrequests of one main request
async def handle_main_request(reqs, args):
    global waiting_time
    global time_start
    global response
    
    for req in reqs:
        waiting_time = waiting_time + np.random.exponential(1.0 / args.request_rate)
        time_elapsed = time.perf_counter() - time_start
        assert waiting_time >= time_elapsed, 'Fail to keep up with the rate of Poisson Distribution' 
        response.append(await post_request_and_get_response(args, req, waiting_time - time_elapsed))

async def run(args, reqs, multi_conversations_range):
    jct = []
    ttft = []
    tbt_no_second_token = []
    tbt_with_second_token = []
    second_token = []
    multi_conversations_range.append(len(reqs))
    coroutines = [
        asyncio.create_task(handle_main_request(
        reqs[multi_conversations_range[i]:multi_conversations_range[i+1]], args))
        for i in range(len(multi_conversations_range) - 1)
    ]
    await asyncio.gather(*coroutines)

    global response 
    for res in response:
        jct.append(res[0])
        ttft.append(res[1])
        tbt_no_second_token.extend(res[2])
        tbt_with_second_token.extend(res[3])
        second_token.append(res[4])
        # print("Res ", res)
    print("average_jct, p99_jct, average_ttft, p99_ttft, average_tbt_no_second_token, p99_tbt_no_second_token, average_tbt_with_second_token, p99_tbt_with_second_token")
    print("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(np.average(jct), np.percentile(jct, 99), np.average(ttft), np.percentile(ttft, 99), np.average(tbt_no_second_token), np.percentile(tbt_no_second_token, 99), np.average(tbt_with_second_token), np.percentile(tbt_with_second_token, 99)))