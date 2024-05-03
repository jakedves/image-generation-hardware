import os

import distinctipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import load_column


if __name__ == '__main__':
    parent_directory = 'output/baseline/'

    dataframe = pd.DataFrame()
    colours = distinctipy.get_colors(2)

    for filename in os.listdir(parent_directory):
        path = parent_directory + filename + '/training_statistics/training_lpips.txt'
        dataframe = pd.concat([dataframe, load_column(path, filename)], axis=1)

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(['mse', 'mae']):
        dataframe[f'{metric}-avg'] = dataframe[[f'{metric}-1', f'{metric}-2', f'{metric}-3']].mean(axis=1)
        dataframe[f'{metric}-error'] = dataframe[[f'{metric}-1', f'{metric}-2', f'{metric}-3']].std(axis=1) / np.sqrt(3)

        colour = colours[i]

        plt.plot(dataframe[f'{metric}-avg'], label=f'${' \\mathcal{L}'}_{'1' if metric == 'mae' else '2'}$', color=colour)
        plt.fill_between(dataframe.index + 1,
                         dataframe[f'{metric}-avg'] - dataframe[f'{metric}-error'],
                         dataframe[f'{metric}-avg'] + dataframe[f'{metric}-error'],
                         color=colour,
                         alpha=0.3)

    size = 'xx-large'
    plt.xlabel('Epochs', fontsize=size)
    plt.ylabel('LPIPS', fontsize=size)
    plt.xlim(0, 200)
    plt.ylim(0.1, 0.6)
    plt.yscale('log')

    plt.title(f'${' \\mathcal{L}'}_1$ vs ${'\\mathcal{L}'}_2$ Loss for Quantised ESPCN (8-bit)', fontsize=size)
    plt.legend(fontsize=size)
    plt.grid(True)

    plt.savefig('mse-vs-mae-log.png', dpi=300)
    print(dataframe)
