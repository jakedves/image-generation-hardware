import os

from subprocess import Popen


def yuv_to_mp4(yuv_path):
    out_name = yuv_path[:-4] if yuv_path.endswith('.yuv') else yuv_path
    out_name = out_name.lower().replace(' ', '_') + '.mp4'

    cmd = [
        'ffmpeg', '-video_size', '3840x2160', '-pixel_format', 'yuv420p', '-framerate', '30', '-i', yuv_path, out_name
    ]

    Popen(cmd, shell=True)


if __name__ == '__main__':
    directory = 'datasets/frame-interpolation/sjtu-4k/'

    for file in os.listdir(directory):
        if file.endswith('.txt'):
            continue

        yuv_to_mp4(directory + file)
