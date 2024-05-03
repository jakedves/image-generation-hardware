import json

import distinctipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    root = 'output/baseline/'
    datatypes = ['fp32', 'int8', 'int6', 'int4', 'int2']
    networks = ['ESPCN', 'MESPCN', 'FESPCN']
    datasets = ['BSD100', 'Urban100', 'Set5', 'Set14']
    filename = 'metrics.json'

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True)
    plt.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.15)
    bar_width = 0.2
    positions = np.arange(len(datatypes))
    colours = distinctipy.get_colors(len(networks), rng=29)

    # xx-large equivalent to latex font when full screen
    font_size = 'x-large'

    for index, dataset_to_visualise in enumerate(datasets):
        if index == 0:
            axis = axes[0, 0]
        elif index == 1:
            axis = axes[0, 1]
        elif index == 2:
            axis = axes[1, 0]
        else:
            axis = axes[1, 1]

        to_plot = {
            'ESPCN': [],
            'MESPCN': [],
            'FESPCN': []
        }

        # for each datatype we want a new table
        for i, datatype in enumerate(datatypes):
            # this loop represents a single table
            df = pd.DataFrame(index=datasets, columns=networks)

            # for row in the table
            for dataset in datasets:
                dataset_lpips_scores = []
                for j, network in enumerate(networks):
                    path = root + datatype + '-' + network

                    row_col_lpips = 0.0
                    runs = ['', '-2', '-3']
                    for run in runs:
                        full_path = path + run + '/'

                        # for this dataset, load each json[dataset] and add it
                        with open(full_path + filename) as file:
                            data = json.load(file)

                        row_col_lpips += float(data['lpips'][dataset])

                    row_col_lpips /= len(runs)
                    dataset_lpips_scores.append("{:.4f}".format(row_col_lpips))

                    if dataset == dataset_to_visualise:
                        to_plot[network].append(row_col_lpips)

                df.loc[dataset] = dataset_lpips_scores

        axis.grid(True, zorder=0)
        axis.bar(positions - bar_width, to_plot['ESPCN'], width=bar_width, color=colours[0], zorder=5)
        axis.bar(positions, to_plot['MESPCN'], width=bar_width, color=colours[1], zorder=5)
        axis.bar(positions + bar_width, to_plot['FESPCN'], width=bar_width, color=colours[2], zorder=5)

        axis.set_xlabel('Precision', fontsize=font_size)
        axis.set_ylabel('LPIPS', fontsize=font_size)
        axis.set_title(f'{dataset_to_visualise}', fontsize=font_size)

        axis.set_xticks(positions)
        axis.set_xticklabels(list(map(lambda x: x.upper(), datatypes)), fontsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)

    figure.legend(networks, loc='lower center', bbox_to_anchor=(0.5, 0), fontsize=font_size, ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.2, bottom=0.12)

    # plt.show()
    plt.savefig(f'lpips_bars-all.png', dpi=300)

        # lt = df.to_latex(index=True, column_format='|l|c|c|c|', caption=f'Test set evaluation on {datatype.upper()} networks', header=True, escape=False)
        # lt = lt.replace('\\toprule', '\\hline')
        # lt = lt.replace('\\midrule', '\\hline')
        # lt = lt.replace('\\bottomrule', '\\hline')
        # print(lt)

