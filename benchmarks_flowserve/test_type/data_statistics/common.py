import json
from typing import Iterable, List, Optional, Tuple
import asyncio

async def post_request_and_get_response(args, req, waiting_time):
    
    await asyncio.sleep(waiting_time)
    
