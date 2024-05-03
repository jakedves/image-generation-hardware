from matplotlib import pyplot as plt
from datetime import datetime
from PIL import Image
import json
import os

from .model_export import torch_export, onnx_export
from .evaluation import evaluate_psnr, evaluate_lpips
from .image_gen import generate_image
from metrics import GPUMetrics, CPUMetrics, total_parameters, model_size
from neural_networks import ModelConfig


DEFAULT_IMAGE = './datasets/super-resolution/Set14/img_005_SRF_2_LR.png'


class DataExport:
    def __init__(self, model, cfg: ModelConfig, student=False):
        prefix = f'int{cfg.bit_width}-' if cfg.quantised else 'fp32-'
        student_tag = '-student' if student else ''
        self.root = f"output/{prefix}{cfg.name}{student_tag}/"

        suffix = 1
        while os.path.exists(self.root):
            suffix += 1
            self.root = f"output/{prefix}{cfg.name}{student_tag}-{suffix}/"

        self.image_directory = self.root + 'images/'
        self.model_directory = self.root + 'models/'
        self.training_statistics_directory = self.root + 'training_statistics/'
        self.plots_directory = self.root + 'plots/'

        dirs = [
            self.root,
            self.image_directory,
            self.model_directory,
            self.training_statistics_directory,
            self.plots_directory
        ]

        for folder in dirs:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.model = model
        self.gpu = GPUMetrics()
        self.cpu = CPUMetrics()

    def export_image(self, path=DEFAULT_IMAGE, filename='final'):
        print("Exporting generated image")
        image = generate_image(self.model, path)
        image_pil = Image.fromarray(image)
        directory = self.root if filename == 'final' else self.image_directory
        image_pil.save(directory + f'{filename}.png')

    def export_training_graph(self, y_label, title, values, filename):
        print(f"Exporting training graph: {title}")
        x_values = range(1, len(values) + 1)

        plt.plot(x_values, values, marker='x', linestyle='-')
        plt.xlabel("Epoch")
        plt.ylabel(y_label)

        plt.title(title)
        plt.grid(True)

        plt.savefig(self.plots_directory + f'{filename}.png')
        plt.close()

    def export_metrics(self):
        print("Exporting metrics")
        filename = self.root + 'metrics.json'
        metrics = {}

        self._append_throughput(metrics)
        self._append_quality(metrics)

        with open(filename, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)

    def _append_throughput(self, metrics):
        metrics['throughput'] = {}

        metrics['throughput']['cpu'] = self.cpu.multi_throughput(self.model)
        metrics['throughput']['gpu'] = self.gpu.multi_throughput(self.model)
        metrics['throughput']['fpga'] = 1.0

    def _append_quality(self, metrics):
        metrics['psnr'] = evaluate_psnr(self.model)
        metrics['lpips'] = evaluate_lpips(self.model)

    def export_models(self, dummy_input_shape, cfg):
        print("Exporting models")
        torch_export(self.model, self.model_directory)
        onnx_export(self.model, dummy_input_shape, cfg, self.model_directory)

    def export_list(self, values, filename):
        with open(self.training_statistics_directory + filename + '.txt', 'w') as file:
            for item in values:
                file.write(str(item) + '\n')

    def export_json(self, dictionary, path):
        with open(self.root + path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)

    def export_configuration(self, model, cfg: ModelConfig):
        export_data = cfg.json().copy()
        params = total_parameters(model)
        size = model_size(model)

        export_data['parameters'] = params
        export_data['network_size'] = '{:.3f}MiB'.format(size)

        self.export_json(export_data, 'configuration.json')
