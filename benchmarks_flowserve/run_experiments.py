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
import subprocess
from io import StringIO

# Configurable parameters
basename = 'end2end_exp_results'
dataset = 'LooGLE' # ['ShareGPT', 'LooGLE', 'ReAct']
configs = {
    'type': 'disagg_layer',
    'num_requests': 4
}
request_rates= [1,2,3] # x-axis 


# Derived parameters
dirname = f'{basename}/{dataset}'

if not os.path.exists(dirname):
    os.makedirs(dirname)

for i, request_rate in enumerate(request_rates):
    command = f'python3 ./main.py --dataset {dataset} --request-rate {request_rate} --num-requests {configs["num_requests"]}'
    print(f'Running command: {command}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    out = out.decode('utf-8')
    csv_data = StringIO(out)
    if i == 0:
        df = pd.read_csv(csv_data)
    else:
        df = pd.concat([df, pd.read_csv(csv_data)])


df.insert(0, 'request_rate', request_rates)
df.to_csv(f'{dirname}/{configs["type"]}.csv', index=False)


# Plotting
os.system('python plot_result.py')
