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

def execute_exp(deploy_type):
    # Configurable parameters
    basename = 'end2end_exp_results'
    dataset = 'LooGLE' # ['ShareGPT', 'LooGLE', 'ReAct']
    configs = {
        'num_requests': 256,
    }
    configs['deploy_type'] = deploy_type
    
    request_rates = []
    req = 1
    while req <= 10:
        request_rates.append(req)
        req = req + 1
    
    if "disagg" in configs['deploy_type']:
        request_rates = [x * 2 for x in request_rates]

    # Derived parameters
    dirname = f'{basename}/{dataset}/{configs["deploy_type"]}'
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for request_rate in request_rates:
        duration = 100 / request_rate
        command = f'python3 main.py --test-type {"open"} --dataset {dataset} --request-rate {request_rate} --num-requests {configs["num_requests"]} '
        print(f'Running command: {configs["deploy_type"]}, {command}')
        os.system(f'{command}')
        time.sleep(5)

if __name__ == "__main__":
    deploy_type = sys.argv[1]
    execute_exp(deploy_type)