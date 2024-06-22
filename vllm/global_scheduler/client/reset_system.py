import vllm.global_scheduler.entrypoints_config as cfg
from typing import Dict, Set, List, Optional
import requests
import time
def post_request(api_url, request_dict: Optional[Dict] = {}):
    headers = {"User-Agent": "Test Client"}
    resp = requests.post(api_url, headers=headers, json=request_dict)
    return resp


def reset_system(host, port):

    creat_comm_api_url = cfg.reset_system_url % (host, port)
    payload = {"reset":"reset"}
    resp = post_request(creat_comm_api_url, payload)
    return resp

resp = reset_system(cfg.eprefill_host, 8082)
