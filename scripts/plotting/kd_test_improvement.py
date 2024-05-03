import json
import os

import distinctipy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def get_axis(index, a):
    if index == 0:
        return a[0, 0]

    if index == 1:
        return a[0, 1]

    if index == 2:
        return a[1, 0]

    return a[1, 1]


def average_on(set_name, path_to_dir):
    """
    Give me a set name (BSD100) and a path (output/baseline/int8-ESPCN)
    and I will return the average value of the LPIPS on all runs of that model
    """
    extensions = ['', '-2', '-3']
    metrics_path = 'metrics.json'

    total = 0
    for extension in extensions:
        if not os.path.isdir(path_to_dir + extension):
            return False

        with open(path_to_dir + extension + f'/{metrics_path}') as f:
            total += json.load(f)['lpips'][set_name]

    return total / 3.0


if __name__ == '__main__':
    path = '../../output/'
    kd_path = path + 'knowledge-distillation/'
    base_path = path + 'baseline/'

    test_sets = ['BSD100', 'Urban100', 'Set5', 'Set14']
    alphas = ['0.05', '0.2', '0.3', '0.8']
    bit_width = 'int8'
    bar_width = 0.2

    positions = np.arange(len(alphas))

    font_size = 'x-large'

    colours = distinctipy.get_colors(2, rng=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)

    for i, test_set in enumerate(test_sets):
        axis = get_axis(i, axes)
        axis.grid(True)

        # these should always exist as from baseline
        data_path = base_path + bit_width + '-'
        espcn_avg = average_on(test_set, data_path + 'ESPCN')
        mespcn_avg = average_on(test_set, data_path + 'MESPCN')

        espcn_improvements = []
        mespcn_improvements = []

        for alpha in alphas:
            student_path = kd_path + f'alpha-{alpha}/'

            # find the average on this test set for ESPCN-bit_width + kd (if exists)
            e_path = student_path + 'same-arch/' + bit_width + '-ESPCN-student'
            espcn_student_avg = average_on(test_set, e_path)

            # find the average on this test set for MESPCN-bit_width + kd (if exists)
            m_path = student_path + 'diff-arch/' + bit_width + '-MESPCN-student'
            mespcn_studnt_avg = average_on(test_set, m_path)

            espcn_improvement = ((espcn_student_avg - espcn_avg) / espcn_avg) * 100 if espcn_student_avg else 0
            mespcn_improvement = ((mespcn_studnt_avg - mespcn_avg) / mespcn_avg) * 100 if mespcn_studnt_avg else 0

            espcn_improvements.append(espcn_improvement)
            mespcn_improvements.append(mespcn_improvement)

        # plot here
        axis.axhline(0, color='red', linestyle='-')
        axis.bar(positions - bar_width, espcn_improvements, width=bar_width, color=colours[0], zorder=5)
        axis.bar(positions, mespcn_improvements, width=bar_width, color=colours[1], zorder=5)

        axis.set_xlabel('$\\alpha$', fontsize=font_size)
        axis.set_ylabel('LPIPS Increase (%)', fontsize=font_size)
        axis.set_title(f'{test_set}', fontsize=font_size)

        axis.set_xticks(positions)
        axis.set_xticklabels(list(map(lambda x: x.upper(), alphas)), fontsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)

    handles = [mpatches.Patch(color=colour, label=label) for colour, label in zip(colours, ['ESPCN', 'MESPCN'])]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0), fontsize=font_size, ncol=2)
    plt.subplots_adjust(hspace=0.35, wspace=0.2, bottom=0.12)
    fig.suptitle(f'Quantised Distillation Impact for {bit_width.upper()} Networks', fontsize='xx-large')
    # plt.show()
    plt.savefig(f'lpips-test-set-{bit_width}.png', dpi=300)
