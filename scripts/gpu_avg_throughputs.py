import json
import os

if __name__ == '__main__':
    base_dir = '../output/baseline/'
    datatype = 'fp32'
    networks = ['ESPCN', 'MESPCN', 'FESPCN']
    file = 'metrics.json'

    throughputs = {}
    for network in networks:
        total_128, total_256, total_512 = 0, 0, 0

        folders = [datatype + '-' + network + run for run in ['', '-2', '-3']]
        for folder in folders:
            filename = base_dir + folder + '/' + file

            with open(filename, 'r') as f:
                data = json.load(f)['throughput']['gpu']
                total_128 += float(data['128x128 images'])
                total_256 += float(data['256x256 images'])
                total_512 += float(data['512x512 images'])

        throughputs[network] = {
            '128x128 fps': total_128 / len(folders),
            '256x256 fps': total_256 / len(folders),
            '512x512 fps': total_512 / len(folders)
        }

    with open('../gpu-throughputs.json', 'w') as writeable:
        json.dump(throughputs, writeable, indent=4)


