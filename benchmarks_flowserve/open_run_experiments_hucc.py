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
import pandas as pd
import time

# Configurable parameters
basename = 'end2end_exp_results'
dataset = 'LooGLE' # ['ShareGPT', 'LooGLE', 'ReAct']
configs = {
    'type': 'disagg_ncache',
    'num_requests': 1000,
}

if "disagg" in configs['type']:
    request_rates = [0.2, 0.4, 0.6, 0.8] # x-axis 
else:
    request_rates= [0.1, 0.2, 0.3, 0.4] # x-axis 

# Derived parameters
dirname = f'{basename}/{dataset}/{configs["type"]}'

if not os.path.exists(dirname):
    os.makedirs(dirname)

for i, request_rate in enumerate(request_rates):
    command = f'python3 main.py --test-type {"open"} --dataset {dataset} --num-clients {request_rate} --num-requests {configs["num_requests"]}  --duration 10 '
    print(f'Running command: {configs['type']}, {command}')
    os.system(f'{command}')
    time.sleep(5)
