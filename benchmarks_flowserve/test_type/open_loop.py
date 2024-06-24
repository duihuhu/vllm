from typing import Iterable, List, Optional, Tuple
import asyncio
import numpy as np 
import time
from .common import post_request_and_get_response, dummy_post_request_and_get_response
import math 
import sys

'''
Warning: The code snippet in this file is purely asynchronous and single-threaded.
And that is why we are going wild in using global variables.
Please do not follow the code convention in this file in other settings.
'''
waiting_time = 0
time_start = time.perf_counter()
response = []

# Handle all subrequests of one main request
async def handle_main_request(main_request_id, reqs, args):
    global waiting_time
    global time_start
    global response
    res = None
    for i in range(len(reqs)):
        if i != 0:
            prev_prompt_len = reqs[i-1][-2]
            prev_completion_len = reqs[i-1][-1]
            prev_completion_token_ids = res[-1]
            reqs[i][1][prev_prompt_len:prev_prompt_len+prev_completion_len] = prev_completion_token_ids
        waiting_time = waiting_time + np.random.exponential(1.0 / args.request_rate)
        time_elapsed = time.perf_counter() - time_start
        # if waiting_time < time_elapsed:
        #     print(f"\033[93m Warning: main_request_id {main_request_id} sub_request_id {i}: Poisson violation\033[0m", file=sys.stderr)
        #     print(f"\033[93m Should have been sent at {time_elapsed - waiting_time:.3} seconds ago\033[0m", file=sys.stderr)
        res = await post_request_and_get_response(args, reqs[i], waiting_time - time_elapsed)
        # res = await dummy_post_request_and_get_response(
        #     args, 
        #     reqs[i], 
        #     waiting_time - time_elapsed,
        #     main_request_id=main_request_id,
        #     sub_request_id=i
        # )
        response.append(res)

async def run(args, reqs, multi_conversations_range):
    global time_start
    jct = []
    ttft = []
    tbt_no_second_token = []
    tbt_with_second_token = []
    second_token = []
    multi_conversations_range.append(len(reqs))

    first_few_sessions = math.ceil(args.request_rate)
    coroutines = [
        asyncio.create_task(handle_main_request(
        i, reqs[multi_conversations_range[i]:multi_conversations_range[i+1]], args))
        for i in range(first_few_sessions)
    ]

    # Start global timer
    time_start = time.perf_counter()
    # Kick start the first few sessions
    await asyncio.sleep(0)

    # Deal with the rest of the sessions, add them when the first few sessions cannot meet the Poisson speed
    main_request_id = first_few_sessions
    while main_request_id < len(multi_conversations_range) - 1:
        coroutines.append(asyncio.create_task(handle_main_request(
            main_request_id, 
            reqs[multi_conversations_range[main_request_id]:multi_conversations_range[main_request_id+1]], 
            args
        )))
        main_request_id += 1
        # Sleep for enough time to avoid too many sessions to enter at the same time
        # To avoid oversleep, we subtract 0.1 seconds  
        await asyncio.sleep(waiting_time - (time.perf_counter() - time_start) - 0.1)
        while waiting_time - (time.perf_counter() - time_start) - 0.1 > 0:
            await asyncio.sleep(waiting_time - (time.perf_counter() - time_start) - 0.1)
    
    await asyncio.gather(*coroutines)

    global response 
    assert len(response) == len(reqs), 'Fail to handle all requests'
    for res in response:
        jct.append(res[0])
        ttft.append(res[1])
        tbt_no_second_token.extend(res[2])
        tbt_with_second_token.extend(res[3])
        second_token.append(res[4])
        # print("Res ", res)
    print("average_jct,p99_jct,average_ttft,p99_ttft,average_tbt_no_second_token,p99_tbt_no_second_token,average_tbt_with_second_token,p99_tbt_with_second_token")
    print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(np.average(jct), np.percentile(jct, 99), np.average(ttft), np.percentile(ttft, 99), np.average(tbt_no_second_token), np.percentile(tbt_no_second_token, 99), np.average(tbt_with_second_token), np.percentile(tbt_with_second_token, 99)))