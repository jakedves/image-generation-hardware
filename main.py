from torch.nn import L1Loss, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optimisers

from neural_networks import Networks
from training import train
from export import DataExport
from neural_networks.model_config import ModelConfig
from device import DEVICE
from training import SuperResTrainingConfig

import sys


if __name__ == '__main__':
    """
    Run this script as follows:
    
    python main.py <network_name> <quantised> <bit_width?>
    """
    cmd = len(sys.argv) >= 3
    print("Using cmd arguments" if cmd else "Using default configuration")

    cfg = ModelConfig(
        sys.argv[1] if cmd else "ESPCN",
        sys.argv[2] == "true" if cmd else True,
        (int(sys.argv[3]) if len(sys.argv) > 3 else None) if cmd else 8,
        loss='mse'
    )

    print(f"Network: {cfg.name}")
    print(f"Quantised: {cfg.quantised}")
    print(f"Bit width: {cfg.bit_width}\n" if cfg.quantised else "")
    print(f"Loss function: {cfg.loss}")

    model = Networks.get(cfg).to(DEVICE)

    # ESPCN Specific Settings
    optimiser = optimisers.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ExponentialLR(optimiser, gamma=0.999)

    if model:
        print("Network created...\n")

    exporter = DataExport(model, cfg)
    exporter.export_configuration(model, cfg)

    training_config = SuperResTrainingConfig(
        epochs=200,
        batch_size=16,
        learning_rate=1e-4,
        loss=cfg.loss
    )

    dummy_input_shape = (1, 3, 512, 512)
    print(f"Exporting model with input shape: {dummy_input_shape}")

    psnrs, lpips_scores = train(
        model,
        optimiser,
        training_config,
        scheduler=scheduler,
        exporter=exporter
    )

    exporter.export_image()
    exporter.export_metrics()
    exporter.export_models(dummy_input_shape, cfg)

    psnr_file = 'training_psnr'
    exporter.export_list(list(map(lambda x: float(format(x, '.2f')), psnrs)), psnr_file)
    exporter.export_training_graph(
        "PSNR (db)",
        "PSNR During Training on DIV2K Dataset",
        psnrs,
        psnr_file
    )

    lpips_file = 'training_lpips'
    exporter.export_list(list(map(lambda x: float(format(x, '.4f')), lpips_scores)), lpips_file)
    exporter.export_training_graph(
        "LPIPS",
        "LPIPS Score During Training on DIV2K Dataset",
        lpips_scores,
        lpips_file
    )

