from typing import Iterable, List, Optional, Tuple
import asyncio
import numpy as np 
from .common import post_request_and_get_response, dummy_post_request_and_get_response
async def run(args, reqs):
    jct = []
    ttft = []
    tbt_no_second_token = []
    tbt_with_second_token = []
    second_token = []
    waiting_time = 0
    coroutines = []
    for req in reqs:
        coroutines.append(asyncio.create_task(dummy_post_request_and_get_response(args, req, waiting_time)))
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
    print("average_jct,p99_jct,average_ttft,p99_ttft,average_tbt_no_second_token,p99_tbt_no_second_token,average_tbt_with_second_token,p99_tbt_with_second_token")
    print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(np.average(jct), np.percentile(jct, 99), np.average(ttft), np.percentile(ttft, 99), np.average(tbt_no_second_token), np.percentile(tbt_no_second_token, 99), np.average(tbt_with_second_token), np.percentile(tbt_with_second_token, 99)))