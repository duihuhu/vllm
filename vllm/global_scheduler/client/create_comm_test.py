
import vllm.global_scheduler.entrypoints_config as cfg
from typing import Dict, Set, List, Optional
import requests

def post_request(api_url, request_dict: Optional[Dict] = {}):
    headers = {"User-Agent": "Test Client"}
    resp = requests.post(api_url, headers=headers, json=request_dict)
    return resp


def create_comm():
    uniqe_id_api_url = cfg.comm_uniqe_id_url % (cfg.eprefill_host, cfg.eprefill_port)
    dst_channel = "_".join([str(rank) for rank in [1,2]])
    resp = post_request(uniqe_id_api_url, {"dst_channel": dst_channel})
    return resp
   
resp = create_comm()
print(resp.json())
