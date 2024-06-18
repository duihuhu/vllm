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
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Union

# Configurable parameters
basename = 'end2end_exp_results'
dataset = 'ReAct' # ['ShareGPT', 'LooGLE', 'ReAct']

# Derived parameters
dirname = f'{basename}/{dataset}'

def plot_figure(x_axis: str, y_axis: Union[str, List[str]]):
    plt.figure(figsize=(12, 10))
    for file in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, file)) and '.csv' in file:
            df = pd.read_csv(f'{dirname}/{file}')
            type = file.split('.')[0] + '_'
            plt.plot(df[x_axis], df[y_axis], label = type + df[y_axis].columns)
    
    plt.xlim(0, 110)
    plt.ylim(0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('req/s', fontsize=14)
    plt.ylabel('time (s)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.title(' '.join(y_axis[0].split('_')[1:]), fontsize=20)   
    figure_title = '_'.join(y_axis[0].split('_')[1:])
    plt.savefig(f'{dirname}/{figure_title}.png')
    plt.clf()
    
if __name__ == '__main__':
    for y_axis in [
        ['average_jct', 'p99_jct'], 
        ['average_ttft', 'p99_ttft'], 
        ['average_tbt_no_second_token', 'p99_tbt_no_second_token'], 
        ['average_tbt_with_second_token', 'p99_tbt_with_second_token']
    ]:
        plot_figure(
            x_axis = 'request_rate',
            y_axis = y_axis
        )
    