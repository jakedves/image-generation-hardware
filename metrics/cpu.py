import torch
import time


class CPUMetrics:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size

    def throughput(self, model, width, height):
        print(f"Measuring CPU Throughput for a {width}x{height} image")
        model = model.to('cpu')

        dummy_input = torch.randn(self.batch_size, 3, width, height)

        repetitions = 10
        total_time = 0

        for rep in range(1, repetitions + 1):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            total_time += end - start

        throughput = (repetitions * self.batch_size) / total_time
        print(f"CPU Throughput[{width}x{height} images/s]: {throughput}")
        return throughput

    def multi_throughput(self, model):
        """
        Runs throughput calculations for a range of image sizes, however does
        not vary the batch size
        """
        throughputs = {}
        for length in [128, 256, 512]:
            throughputs[f"{length}x{length} images"] = self.throughput(model, length, length)

        return throughputs
