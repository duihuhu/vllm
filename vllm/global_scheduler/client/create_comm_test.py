
import vllm.global_scheduler.entrypoints_config as cfg
from typing import Dict, Set, List, Optional
import requests

def post_request(api_url, request_dict: Optional[Dict] = {}):
    headers = {"User-Agent": "Test Client"}
    resp = requests.post(api_url, headers=headers, json=request_dict)
    return resp


def create_comm():
    uniqe_id_api_url = cfg.comm_uniqe_id_url % (cfg.eprefill_host, cfg.eprefill_port)
    dst_channel = "_".join([str(rank) for rank in [0,1]])
    resp = post_request(uniqe_id_api_url, {"dst_channel": dst_channel})
    
    creat_comm_api_url = cfg.create_comm_url % (cfg.edecode_host, cfg.edecode_port)
    src_channel =  "_".join([str(rank) for rank in [2,3]])
    payload = {}
    payload['nccl_id'] = resp.json()
    payload['dst_channel'] = src_channel

    resp = post_request(creat_comm_api_url, payload)
    return resp
   
resp = create_comm()
print(resp.json())
