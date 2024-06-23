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
        'num_requests': 1000,
    }
    configs['deploy_type'] = deploy_type

    if "disagg" in configs['deploy_type']:
        request_rates = [0.2, 0.4, 0.6, 0.8] # x-axis 
    else:
        request_rates= [0.1, 0.2, 0.3, 0.4] # x-axis 

    request_rates = []
    req = 0.5
    while req <= 20:
        request_rates.append(req)
        req = req + 0.5
    
    if "disagg" in configs['deploy_type']:
        request_rates = [x * 2 for x in request_rates]

    # Derived parameters
    dirname = f'{basename}/{dataset}/{configs["deploy_type"]}'
    
    # Derived parameters
    dirname = f'{basename}/{dataset}/{configs["deploy_type"]}'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for request_rate in request_rates:
        duration = 100 / request_rate
        command = f'python3 main.py --test-type {"open"} --dataset {dataset} --request-rate {request_rate} --num-requests {configs["num_requests"]}  --duration {duration} '
        print(f'Running command: {configs["deploy_type"]}, {command}')
        os.system(f'{command}')
        time.sleep(5)

if __name__ == "__main__":
    deploy_type = sys.argv[1]
    execute_exp(deploy_type)