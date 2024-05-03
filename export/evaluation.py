from util.dataset_names import BSD100, Set5, Set14, Urban100
from util import SuperResolutionDataset
from metrics import average_psnr, average_lpips

from torch.utils.data import DataLoader
import torch


def evaluate_psnr(model):
    results = {}

    for dataset in [BSD100, Set5, Set14, Urban100]:
        data = SuperResolutionDataset(dataset, patch_size=None)
        eval_loader = DataLoader(data, batch_size=1, shuffle=False)

        model.eval()
        with torch.no_grad():
            avg_psnr = average_psnr(eval_loader, model)

        results[dataset] = float("{:.2f}".format(avg_psnr))

    return results


def evaluate_lpips(model):
    results = {}

    for dataset in [BSD100, Set5, Set14, Urban100]:
        data = SuperResolutionDataset(dataset, patch_size=None)
        eval_loader = DataLoader(data, batch_size=1, shuffle=False)

        model.eval()
        with torch.no_grad():
            avg_lpips = average_lpips(eval_loader, model)

        results[dataset] = float("{:.4f}".format(avg_lpips))

    return results
