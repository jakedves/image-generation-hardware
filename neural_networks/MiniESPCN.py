from brevitas_examples.super_resolution.models import FloatESPCN, QuantESPCN
from brevitas_examples.super_resolution.models.common import CommonIntWeightPerChannelQuant
from brevitas.nn.quant_layer import WeightQuantType
from torch import Tensor
import torch


class MiniESPCN(FloatESPCN):
    def __init__(self, upscale_factor: int = 2, num_channels: int = 3):
        super().__init__(upscale_factor=upscale_factor, num_channels=num_channels)
        self.conv2 = None
        self.bn2 = None

    def forward(self, inp: Tensor):
        x = torch.relu(inp)  # Adding for finn-onnx compatability
        x = self.relu(self.bn1(self.conv1(x)))

        # remove the conv2, bn2 layers
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pixel_shuffle(self.conv4(x))
        x = self.out(x)  # To mirror quant version
        return x


class MiniQuantESPCN(QuantESPCN):
    def __init__(self,
                 upscale_factor: int = 2,
                 num_channels: int = 3,
                 weight_bit_width: int = 4,
                 act_bit_width: int = 4,
                 acc_bit_width: int = 32,
                 weight_quant: WeightQuantType = CommonIntWeightPerChannelQuant):
        super().__init__(upscale_factor=upscale_factor,
                         num_channels=num_channels,
                         weight_bit_width=weight_bit_width,
                         act_bit_width=act_bit_width,
                         acc_bit_width=acc_bit_width,
                         weight_quant=weight_quant)
        self.conv2 = None
        self.bn2 = None

    def forward(self, inp: Tensor):
        x = torch.relu(inp)  # Adding for finn-onnx compatability
        x = self.relu(self.bn1(self.conv1(x)))

        # remove the conv2, bn2 layers
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pixel_shuffle(self.conv4(x))
        x = self.out(x)  # To mirror quant version
        return x
