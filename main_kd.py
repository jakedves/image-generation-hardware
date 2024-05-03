from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optimisers

from neural_networks import Networks
from training import train
from export import DataExport
from neural_networks.model_config import ModelConfig
from device import DEVICE
from training import SuperResTrainingConfig

if __name__ == '__main__':
    """
    Run this script as follows:

    python main.py <network_name> <quantised> <bit_width?>
    """

    teacher_config = ModelConfig(
        "ESPCN",
        True,
        8,
        loss='mse'
    )

    print("Knowledge Distillation, training teacher...")

    teacher_model = Networks.get(teacher_config).to(DEVICE)

    # ESPCN Specific Settings
    t_optimiser = optimisers.Adam(teacher_model.parameters(), lr=1e-4, weight_decay=1e-4)
    t_scheduler = ExponentialLR(t_optimiser, gamma=0.999)

    if teacher_model:
        print("Networks created...\n")

    t_exporter = DataExport(teacher_model, teacher_config)
    t_exporter.export_configuration(teacher_model, teacher_config)

    t_training_config = SuperResTrainingConfig(
        epochs=200,
        batch_size=16,
        learning_rate=1e-4,
        loss=teacher_config.loss
    )

    dummy_input_shape = (1, 3, 512, 512)
    print(f"Exporting model with input shape: {dummy_input_shape}")

    psnrs, lpips_scores = train(
        teacher_model,
        t_optimiser,
        t_training_config,
        scheduler=t_scheduler,
        exporter=t_exporter
    )

    t_exporter.export_image()
    t_exporter.export_metrics()
    t_exporter.export_models(dummy_input_shape, teacher_config)

    psnr_file = 'training_psnr'
    t_exporter.export_list(list(map(lambda x: float(format(x, '.2f')), psnrs)), psnr_file)
    t_exporter.export_training_graph(
        "PSNR (db)",
        "PSNR During Training on DIV2K Dataset",
        psnrs,
        psnr_file
    )

    lpips_file = 'training_lpips'
    t_exporter.export_list(list(map(lambda x: float(format(x, '.4f')), lpips_scores)), lpips_file)
    t_exporter.export_training_graph(
        "LPIPS",
        "LPIPS Score During Training on DIV2K Dataset",
        lpips_scores,
        lpips_file
    )

    # MARK : At this point the teacher is trained
    teacher_model = teacher_model.to(DEVICE)
    teacher_model.eval()

    i = 1
    for name in ["ESPCN"]:
        for bw in [6, 4, 2]:
            if name == "ESPCN" and bw == 8:
                continue

            student_cfg = ModelConfig(
                name,
                True,
                bw,
                loss='mse'
            )

            student_model = Networks.get(student_cfg).to(DEVICE)
            print(f"Training student {i}")
            print(f"Network: {student_cfg.name}")
            print(f"Quantised: {student_cfg.quantised}")
            print(f"Bit width: {student_cfg.bit_width}\n" if student_cfg.quantised else "")
            print(f"Loss function: {student_cfg.loss}")
            i += 1

            optimiser = optimisers.Adam(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = ExponentialLR(optimiser, gamma=0.999)

            exporter = DataExport(student_model, student_cfg, True)
            exporter.export_configuration(student_model, student_cfg)

            batch_size = 8

            training_config = SuperResTrainingConfig(
                epochs=200,
                batch_size=batch_size,
                learning_rate=1e-4,
                loss=student_cfg.loss
            )

            psnrs, lpips_scores = train(
                student_model,
                optimiser,
                training_config,
                scheduler=scheduler,
                exporter=exporter,
                teacher=teacher_model,
                accum_batches=int(16 / batch_size)
            )

            exporter.export_image()
            exporter.export_metrics()
            exporter.export_models(dummy_input_shape, student_cfg)

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
