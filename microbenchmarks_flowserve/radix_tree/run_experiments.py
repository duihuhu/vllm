'''
We assume the following result folder structure:

basename  
    config1.csv
    config2.csv

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
basename = 'radix_tree_exp_results'
configs = {
    'block_size': 16,
    'num_blocks': 16,
}
cache_ratio = [i / 10 for i in range(11)] # x-axis 


# Derived parameters
dirname = f'{basename}/'

if not os.path.exists(dirname):
    os.makedirs(dirname)

for i in range(len(cache_ratio)):
    command = f'python3 ./main.py --block_size {configs["block_size"]} --num_blocks {configs["num_blocks"]} --cache-ratio {cache_ratio[i]}'
    print(f'Running command: {command}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    out = out.decode('utf-8')
    csv_data = StringIO(out)
    if i == 0:
        df = pd.read_csv(csv_data)
    else:
        df = pd.concat([df, pd.read_csv(csv_data)])


df.insert(0, 'cache_ratio', cache_ratio)
df.to_csv(f'{dirname}/{configs["block_size"]}_{configs["num_blocks"]}.csv', index=False)


# Plotting
os.system('python plot_result.py')
