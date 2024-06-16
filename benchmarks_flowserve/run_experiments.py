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

# Configurable parameters
basename = 'end2end_exp_results'
dataset = 'ReAct' # ['ShareGPT', 'LooGLE', 'ReAct']
configs = {
    'type': 'disagg_layer',
    'num_requests': 256
}
request_rates= [3.2, 6.4, 12.8, 25.6, 51.2, 102.4] # x-axis

# Derived parameters
dirname = f'{basename}/{dataset}'

if not os.path.exists(dirname):
    os.makedirs(dirname)

# Run experiments and record results
output_file = f'{dirname}/{configs["type"]}_{configs["num_requests"]}.csv'
temp_file = f'{dirname}/temp.csv'

for i, request_rate in enumerate(request_rates):
    command = f'python3 ./main.py --dataset {dataset} --request-rate {request_rate} --num-requests {configs["num_requests"]}'
    print(f'Running command: {command} > {temp_file}')
    os.system(f'{command} > {temp_file}')
    if i == 0:
        output_csv = pd.read_csv(temp_file)
        output_csv['request_rate'] = request_rate
    else:
        temp_csv = pd.read_csv(temp_file)
        temp_csv['request_rate'] = request_rate
        output_csv = pd.concat([output_csv, temp_csv])
    
output_csv.to_csv(output_file, index=False)
