import torch

from util import downsample


def psnr(a, b):
    mse = torch.mean((a - b) ** 2)

    if mse == 0:
        print('MSE == 0')
        return float('inf')

    max_value = 1.0  # the .ToTensor transformation maps values from 255.0 -> 1.0
    value = 20 * torch.log10(max_value / torch.sqrt(mse))

    return value.item()


def average_psnr(targets, generate):
    """
    Given a dataset loader, and an image-generating function,
    returns the average PSNR of the image-generating function on that dataset
    """
    values = []

    for target in targets:
        x = downsample(target)
        output = generate(x)
        values.append(psnr(output, target))

    return sum(values) / len(values)
