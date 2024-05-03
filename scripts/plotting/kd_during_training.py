import os

import distinctipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


from util import load_column


if __name__ == '__main__':
    original_directory = 'output/baseline/'
    kd_directory = 'output/knowledge-distillation/'

    df = pd.DataFrame()

    alpha = '0.8'
    metric = 'lpips'
    network = 'MESPCN'  # cannot be FESPCN

    variables = ['int6', 'int4', 'int2'] + ([] if network == 'ESPCN' else ['int8'])
    colours = distinctipy.get_colors(len(variables), rng=29)


    data_file = f'/training_statistics/training_{metric}.txt'
    full_student_directory = kd_directory + f'alpha-{alpha}/' + f'{"same" if network == "ESPCN" else "diff"}-arch/'

    for directory in [original_directory, full_student_directory]:
        for filename in os.listdir(directory):
            if network not in filename or 'fp32' in filename or ('int8' in filename and network == 'ESPCN'):
                continue

            path = directory + filename + data_file
            df = pd.concat([df, load_column(path, filename)], axis=1)

    plt.figure(figsize=(10, 6))

    num_averages = 3

    for i, variable in enumerate(variables):
        og_prefix = variable + '-' + network
        student_prefix = og_prefix + '-student'

        for prefix in [student_prefix, og_prefix]:
            df[f'{prefix}-avg'] = df[[f'{prefix}', f'{prefix}-2', f'{prefix}-3']].mean(axis=1)
            df[f'{prefix}-err'] = df[[f'{prefix}', f'{prefix}-2', f'{prefix}-3']].std(axis=1) / np.sqrt(num_averages)
            df[f'{prefix}-relerr'] = df[f'{prefix}-err'] / df[f'{prefix}-avg']

        df[f'{og_prefix}-change'] = ((df[f'{student_prefix}-avg'] / df[f'{og_prefix}-avg']) - 1.0) * 100.0

        df[f'{og_prefix}-change-err'] = (df[f'{student_prefix}-relerr'] + df[f'{og_prefix}-relerr'])
        df[f'{og_prefix}-change-err'] = df[f'{og_prefix}-change-err'] * df[f'{og_prefix}-change'] * 10.0

        colour = colours[i]

        plt.plot(df.index + 1,
                 df[f'{og_prefix}-change'],
                 label=variable.upper(),
                 color=colour)

        plt.fill_between(
            df.index + 1,
            df[f'{og_prefix}-change'] - df[f'{og_prefix}-change-err'],
            df[f'{og_prefix}-change'] + df[f'{og_prefix}-change-err'],
            color=colour,
            alpha=0.3
        )

    def formatt(x, pos):
        return f'{int(x)}%'

    formatter = FuncFormatter(formatt)
    plt.gca().yaxis.set_major_formatter(formatter)

    size = 'xx-large'
    plt.axhline(0, color='red', linestyle='-')
    plt.xlabel('Epochs', fontsize=size)
    plt.ylabel(metric.upper() + ' Change', fontsize=size)
    plt.xlim(0, 200)
    plt.ylim(-25, 25)

    plt.title(f'{"ESPCN to MESPCN" if network == "MESPCN" else "ESPCN"} with INT8 Quantised Distillation ($\\alpha = {alpha}$)', fontsize=size)
    plt.legend(fontsize=size)
    plt.grid(True)
    # plt.show()

    plt.savefig(f'kd-train-diff-a{alpha}.png', dpi=300)
    print(df)
