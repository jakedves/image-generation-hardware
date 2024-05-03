import torch.nn.functional as F


def downsample(image_tensor):
    return F.interpolate(image_tensor, scale_factor=0.5, mode='bicubic', align_corners=False)
