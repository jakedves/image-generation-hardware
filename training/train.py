import torch

from tqdm import tqdm

from metrics import average_psnr, average_lpips
from util.downsample import downsample

from device import DEVICE

LOSS = {
    'mse': torch.nn.MSELoss(),
    'mae': torch.nn.L1Loss()
}


def train(model, optimiser, t_cfg, scheduler, exporter, teacher=None, accum_batches=1):
    psnr_per_epoch = []
    lpips_per_epoch = []

    loss = LOSS[t_cfg.loss].to(DEVICE)

    for epoch in range(1, t_cfg.epochs + 1):
        progress_bar = tqdm(
            total=len(t_cfg.training_set),
            desc=f'Epoch {epoch}',
            unit=f'image batch of {t_cfg.batch_size}'
        )

        model.train()
        optimiser.zero_grad()

        for iteration, targets in enumerate(t_cfg.training_set):
            x_batch = downsample(targets)

            outputs = model(x_batch)

            if teacher:
                t_outputs = teacher(x_batch)
                loss_val = 0.95 * loss(outputs, targets) + 0.05 * loss(outputs, t_outputs)
            else:
                loss_val = loss(outputs, targets)

            loss_val = loss_val / accum_batches

            # store gradients in the tensors that hold the parameters of model
            loss_val.backward()

            # iterate over parameters and use stored value to update them
            if (iteration + 1) % accum_batches == 0:
                optimiser.step()
                optimiser.zero_grad()

            progress_bar.update(1)

        progress_bar.close()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            psnr = average_psnr(t_cfg.validation_set, model)
            psnr_per_epoch.append(psnr)
            print(f"Epoch: {epoch}/{t_cfg.epochs}\nAverage PSNR: {psnr}")

            lpips = average_lpips(t_cfg.validation_set, model)
            lpips_per_epoch.append(lpips)
            print(f"Average LPIPS: {lpips}\n")

        exporter.export_image(path='./datasets/super-resolution/Set14/img_011_SRF_2_LR.png', filename=epoch)

    return psnr_per_epoch, lpips_per_epoch
