

import matplotlib.pyplot as plt
from distinctipy import distinctipy
from matplotlib.ticker import FuncFormatter

if __name__ == '__main__':
    espcn = [39044, 71761, 80473, 92468]
    mespcn = [27142, 38349, 43573, 45901]
    fespcn = [37004, 71761, 80473, 92485]

    x = ['INT2', 'INT4', 'INT6', 'INT8']

    colours = distinctipy.get_colors(3, rng=5)

    plt.plot(x, espcn, label='ESPCN', color=colours[0], zorder=100)
    plt.plot(x, mespcn, label='MESPCN', color=colours[1])
    plt.plot(x, fespcn, label='FESPCN', color=colours[2])

    font = 'x-large'
    plt.xlabel('Bit Width', fontsize=font)
    plt.ylabel('LUTs', fontsize=font)
    plt.title('Bit Width vs Generated Hardware Size', fontsize=font)
    plt.legend(fontsize=font)
    plt.grid(True)

    def format_ticks(x, pos):
        return f'{int(x/1000)}K'

    formatter = FuncFormatter(format_ticks)
    plt.gca().yaxis.set_major_formatter(formatter)

    # plt.show()
    plt.savefig('LUTs.pdf', dpi=300)
