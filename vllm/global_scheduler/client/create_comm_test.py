import vllm.global_scheduler.entrypoints_config as cfg
from typing import Dict, Set, List, Optional
import requests
import time
import argparse
def post_request(api_url, request_dict: Optional[Dict] = {}):
    headers = {"User-Agent": "Test Client"}
    resp = requests.post(api_url, headers=headers, json=request_dict)
    return resp


def create_comm(eprefill_host, eprefill_port, prefill_rank, edecode_host, edecode_port, decode_rank, worker_type):

    print(f"prefill: {eprefill_host}:{eprefill_port} {prefill_rank}. decode: {edecode_host}:{edecode_port} {decode_rank}")
    # send a request to prefill node.
    uniqe_id_api_url = cfg.comm_uniqe_id_url % (eprefill_host, eprefill_port)
    dst_channel = "_".join([str(rank) for rank in decode_rank])
    resp = post_request(uniqe_id_api_url, {"dst_channel": dst_channel, "worker_type":worker_type})
    if resp.status_code != 200:
        print(f"Error: Request to prefill node for {worker_type} failed with status code {resp.status_code}. Response content: {resp.text}")
        return None
    # send a request to decode node. 
    creat_comm_api_url = cfg.create_comm_url % (edecode_host, edecode_port)
    src_channel =  "_".join([str(rank) for rank in prefill_rank])
    payload = {}
    payload['nccl_id'] = resp.json()
    payload['dst_channel'] = src_channel
    payload['worker_type'] = worker_type
    print("payload ", payload)
    resp = post_request(creat_comm_api_url, payload)
    if resp.status_code != 200:
        print(f"Error: Request to decode node for {worker_type} failed with status code {resp.status_code}. Response content: {resp.text}") 
        return None
    return resp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-host", type=str, default="localhost")
    parser.add_argument("--prefill-port", type=int, default=8082)
    parser.add_argument("--decode-host", type=str, default="localhost") 
    parser.add_argument("--decode-port", type=str, default=8083)
    parser.add_argument("--prefill-rank", nargs='+', type=int)
    parser.add_argument("--decode-rank", nargs='+', type=int)
    
    args = parser.parse_args()

    resp = create_comm(args.prefill_host, args.prefill_port, args.prefill_rank, args.decode_host, args.decode_port, args.decode_rank, "sender")
    resp = create_comm(args.prefill_host, args.prefill_port, args.prefill_rank, args.decode_host, args.decode_port, args.decode_rank, "recv")

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