import lpips
from util import downsample
from device import DEVICE

func = lpips.LPIPS(net='alex').to(DEVICE)


def lpips_score(a, b):
    return func(a, b).item()


def average_lpips(targets, generate):
    """
    Given a dataset loader, and an image-generating function,
    returns the average PSNR of the image-generating function on that dataset
    """
    values = []

    for target in targets:
        x = downsample(target)
        output = generate(x)
        values.append(lpips_score(output, target))

    return sum(values) / len(values)
