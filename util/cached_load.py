from functools import cache

from PIL import Image
from torchvision import transforms


@cache
def cached_image_load(image_path):
    return transforms.ToTensor()(Image.open(image_path))
