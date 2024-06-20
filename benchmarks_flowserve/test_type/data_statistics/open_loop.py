from typing import Iterable, List, Optional, Tuple
import asyncio
import numpy as np 
from .common import post_request_and_get_response
async def run(args, reqs):
    
    waiting_time = 0
    coroutines = []
    for req in reqs:
        coroutines.append(asyncio.create_task(post_request_and_get_response(args, req, waiting_time)))
        interval = np.random.exponential(1.0 / args.request_rate)
        waiting_time = waiting_time + interval
    response = await asyncio.gather(*coroutines)