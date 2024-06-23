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
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Union

# Configurable parameters
basename = 'radix_tree_exp_results'

# Derived parameters
dirname = f'{basename}/'

def plot_figure(x_axis: str, y_axis: Union[str, List[str]]):
    plt.figure(figsize=(12, 10))
    for file in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, file)) and '.csv' in file:
            df = pd.read_csv(f'{dirname}/{file}')
            type = file.split('.')[0] + '_'
            plt.plot(df[x_axis], df[y_axis], label = type + df[y_axis].columns)
    
    plt.xlim(0)
    plt.ylim(0)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('cache ratio', fontsize=24)
    plt.ylabel('time (us)', fontsize=24)
    plt.legend()
    plt.grid(True)
    plt.title(' '.join(y_axis[0].split('_')[1:]), fontsize=24)   
    figure_title = '_'.join(y_axis[0].split('_')[1:])
    plt.savefig(f'{dirname}/{figure_title}.png')
    plt.clf()
    
if __name__ == '__main__':
    for y_axis in [
        ['avg_insert_time', 'avg_match_time'],
    ]:
        plot_figure(
            x_axis = 'cache_ratio',
            y_axis = y_axis
        )
    