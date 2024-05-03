from PIL import Image

import os

if __name__ == '__main__':
    directory = 'datasets/super-resolution/DIV2K/validation/'

    for filename in os.listdir(directory):
        print(filename)

        image = Image.open(directory + filename)

        downsampled = image.resize((image.width // 2, image.height // 2), Image.BICUBIC)

        downsampled.save(directory + 'processed/' + filename[:-4] + "_LR.png")
        image.save(directory + 'processed/' + filename[:-4] + "_HR.png")
