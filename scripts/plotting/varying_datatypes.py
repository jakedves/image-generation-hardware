import os

import distinctipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import load_column


if __name__ == '__main__':
    parent_directory = 'output/baseline/'

    dataframe = pd.DataFrame()
    colours = distinctipy.get_colors(5)

    lines = ['fp32', 'int8', 'int6', 'int4', 'int2']
    metric = 'lpips'
    network = 'FESPCN'

    for filename in os.listdir(parent_directory):
        path = parent_directory + filename + f'/training_statistics/training_{metric}.txt'
        dataframe = pd.concat([dataframe, load_column(path, filename)], axis=1)

    plt.figure(figsize=(10, 6))

    num_averages = 3

    for i, datatype in enumerate(lines):
        folder_prefix = datatype + '-' + network
        dataframe[f'{folder_prefix}-avg'] = dataframe[[f'{folder_prefix}', f'{folder_prefix}-2', f'{folder_prefix}-3']].mean(axis=1)
        dataframe[f'{folder_prefix}-error'] = dataframe[[f'{folder_prefix}', f'{folder_prefix}-2', f'{folder_prefix}-3']].std(axis=1) / np.sqrt(num_averages)

        colour = colours[i]

        plt.plot(dataframe[f'{folder_prefix}-avg'], label=datatype.upper(), color=colour)
        plt.fill_between(dataframe.index + 1,
                         dataframe[f'{folder_prefix}-avg'] - dataframe[f'{folder_prefix}-error'],
                         dataframe[f'{folder_prefix}-avg'] + dataframe[f'{folder_prefix}-error'],
                         color=colour,
                         alpha=0.3)

    size = 'xx-large'
    plt.xlabel('Epochs', fontsize=size)
    plt.ylabel(metric.upper(), fontsize=size)
    plt.xlim(0, 200)
    plt.ylim(0.1, 0.3)

    plt.title(f'Varying Bit-Width for {network}', fontsize=size)
    plt.legend(fontsize=size)
    plt.grid(True)
    plt.show()

    # plt.savefig('kd-train-a0.2.png', dpi=300)
    print(dataframe)
