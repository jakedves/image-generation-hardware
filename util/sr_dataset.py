import torch
from torch.utils.data import Dataset

import os

from device import DEVICE
from .cached_load import cached_image_load


class SuperResolutionDataset(Dataset):
    def __init__(self, name, debug=False, scale_factor=2, patch_size=512):
        self.path = './datasets/super-resolution/' + name
        self.scale_factor = scale_factor
        self.patch_size = patch_size

        # Remove '_HR.png' tag and remove duplicates so each pair is considered once
        self.image_file_names = list(set(map(lambda s: s[:-7], os.listdir(self.path))))

        self.unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        if debug:
            print(self.image_file_names)

    def __len__(self):
        return len(self.image_file_names)

    def patch(self, high_res):
        channels, high_h, high_w = high_res.size()

        # sample a random patch from the image, keeping bounds in mind
        x = torch.randint(0, high_h - self.patch_size, (1,))
        y = torch.randint(0, high_w - self.patch_size, (1,))

        high_res = high_res[:, x:x + self.patch_size, y:y + self.patch_size]

        return high_res

    def __getitem__(self, idx):
        image_name = self.path + '/' + self.image_file_names[idx]

        high_res = cached_image_load(image_name + '_HR.png')
        high_res = high_res if self.patch_size is None else self.patch(high_res)

        return high_res.to(DEVICE)
