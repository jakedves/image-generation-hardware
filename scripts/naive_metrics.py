import json

from util import SuperResolutionDataset
from util.dataset_names import Set5, Set14, BSD100, Urban100
from metrics import average_psnr, average_lpips

from torch.utils.data import DataLoader
import torch.nn.functional as F


if __name__ == '__main__':

    for method in ["nearest", "bilinear", "bicubic"]:
        psnrs = {}
        lpipss = {}

        for dataset_name in [BSD100, Set5, Set14, Urban100]:
            dataset = SuperResolutionDataset(dataset_name, patch_size=None)
            dataloader = DataLoader(dataset, batch_size=1)

            corners = False if not method == 'nearest' else None

            psnr = average_psnr(
                dataloader,
                lambda x: F.interpolate(x, scale_factor=2, mode=method, align_corners=corners)
            )

            lpips = average_lpips(
                dataloader,
                lambda x: F.interpolate(x, scale_factor=2, mode=method, align_corners=corners)
            )

            psnrs[dataset_name] = psnr
            lpipss[dataset_name] = lpips
            print(f"PSNR for {dataset_name} with {method} interpolation is {psnr}")
            print(f"LPIPS for {dataset_name} with {method} interpolation is {lpips}")

        with open(f"{method}-psnr.json", "w") as json_file:
            json.dump(psnrs, json_file)

        with open(f"{method}-lpips.json", "w") as json_file2:
            json.dump(lpipss, json_file2)
