import torch

from device import DEVICE


class GPUMetrics:
    def __init__(self, max_batch_size=4):
        self.max_batch_size = max_batch_size

    def throughput(self, model, width, height):
        """
            Functionality to measure GPU throughput correctly, provided by Geifman (2023)

            Considerations:
              1. Memory should be pre-allocated on the GPU
              2. GPU should be in an activate state (not off due to restart time)
              3. CUDA is asynchronous in PyTorch, use the PyTorch timing mechanisms
              4. OPTIMAL_BATCH_SIZE is found by maxing out the GPU memory using manual binary search
              5. Take an average of a large batch

            The Correct Way to Measure Inference Time of Deep Neural Networks, Ammon Geifman,
            May 1st, 2023 [Online]. Accessed from: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
        """
        print(f"Measuring GPU throughput for a {width}x{height} image")
        model = model.to(DEVICE)

        dummy_input = torch.randn(self.max_batch_size, 3, width, height).to(DEVICE)

        repetitions = 500
        total_time = 0

        # Warm up the GPU
        for _ in range(10):
            _ = model(dummy_input)

        # Start measurements
        with torch.no_grad():
            for rep in range(repetitions):
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()

                current_time = starter.elapsed_time(ender) / 1000.0
                total_time += current_time

        throughput = (repetitions * self.max_batch_size) / total_time

        print(f"GPU Throughput[{width}x{height} images/s]: {throughput}")
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

