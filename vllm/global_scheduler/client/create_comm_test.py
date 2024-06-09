import vllm.global_scheduler.entrypoints_config as cfg
from typing import Dict, Set, List, Optional
import requests
import time
def post_request(api_url, request_dict: Optional[Dict] = {}):
    headers = {"User-Agent": "Test Client"}
    resp = requests.post(api_url, headers=headers, json=request_dict)
    return resp


def create_comm(eprefill_port, prefill_rank, edecode_port, decode_rank, worker_type):
    uniqe_id_api_url = cfg.comm_uniqe_id_url % (cfg.eprefill_host, eprefill_port)
    dst_channel = "_".join([str(rank) for rank in decode_rank])
    resp = post_request(uniqe_id_api_url, {"dst_channel": dst_channel, "worker_type":worker_type})
    
    creat_comm_api_url = cfg.create_comm_url % (cfg.edecode_host, edecode_port)
    src_channel =  "_".join([str(rank) for rank in prefill_rank])
    payload = {}
    payload['nccl_id'] = resp.json()
    payload['dst_channel'] = src_channel
    payload['worker_type'] = worker_type
    print("payload ", payload)
    resp = post_request(creat_comm_api_url, payload)
    return resp

resp = create_comm(8082,[0,1],8083,[2,3], "sender")
resp = create_comm(8082,[0,1],8083,[2,3], "recv")

#resp = create_comm(8082,[0,1],8084,[4,5])
'''
resp = create_comm(8082,[0,1],8083,[4,5])
time.sleep(1)
resp = create_comm(8082,[0,1],8083,[4,5])
time.sleep(1)
#resp = create_comm(8082,[0,1],8083,[4,5])
#time.sleep(1)
resp = create_comm(8085,[2,3],8083,[4,5])
time.sleep(1)
resp = create_comm(8085,[2,3],8083,[4,5])
time.sleep(1)
#resp = create_comm(8085,[2,3],8083,[4,5])
#time.sleep(1)
resp = create_comm(8082,[0,1],8084,[6,7])
time.sleep(1)
resp = create_comm(8082,[0,1],8084,[6,7])
time.sleep(1)
#resp = create_comm(8082,[0,1],8084,[6,7])
#time.sleep(1)
resp = create_comm(8085,[2,3],8084,[6,7])
time.sleep(1)
resp = create_comm(8085,[2,3],8084,[6,7])
time.sleep(1)
#resp = create_comm(8085,[2,3],8084,[6,7])
#time.sleep(1)
'''