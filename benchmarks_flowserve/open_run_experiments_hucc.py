'''
We assume the following result folder structure:


basename  
    dataset1
        config1.csv
        config2.csv
        ...
    dataset2
        ...
    dataset3
        ...
    ...

In each of the .csv file, each column represents a different metric, 
and each row represents a different experiment configuration.

Each folder represents a figure that compares  the performance of the baselines 
and our method. Each .csv file represents one line in the figure.
'''
import os
import time
import sys

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

def reset_gs(host, port):
    creat_comm_api_url = cfg.reset_gs_url % (host, port)
    payload = {"reset":"reset"}
    resp = post_request(creat_comm_api_url, payload)
    return resp

def execute_exp(deploy_type, is_cache):
    # Configurable parameters
    basename = 'end2end_exp_results'
    # dataset = 'ReAct' # ['ShareGPT', 'LooGLE', 'ReAct']
    datasets = ['ReAct', 'LooGLE', 'ShareGPT']

    configs = {
        'num_requests': 512,
    }
    configs['deploy_type'] = deploy_type
    
    request_rates = []

    req = 6
    while req <= 10:
        request_rates.append(req)
        req = req + 2
    
    if "disagg" in configs['deploy_type']:
        request_rates = [x * 2 for x in request_rates]

    for dataset in datasets:
        # Derived parameters
        dirname = f'{basename}/{dataset}/{configs["deploy_type"]}'
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for request_rate in request_rates:
            command = f'python3 main.py --test-type {"open"} --dataset {dataset} --request-rate {request_rate} --num-requests {configs["num_requests"]} '
            print(f'Running command: {configs["deploy_type"]}, {command}')
            os.system(f'{command}')
            if is_cache > 0:
                resp = reset_system(cfg.eprefill_host, 8082)
                resp = reset_system(cfg.eprefill_host, 8083)
                resp = reset_system(cfg.eprefill_host, 8084)

            time.sleep(5)

if __name__ == "__main__":
    deploy_type = sys.argv[1]
    is_cache = int(sys.argv[2])
    execute_exp(deploy_type, is_cache)