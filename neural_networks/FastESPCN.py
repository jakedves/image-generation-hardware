from brevitas_examples.super_resolution.models import FloatESPCN, QuantESPCN
from brevitas import nn as qnn
from torch import nn

from brevitas_examples.super_resolution.models.common import CommonIntWeightPerChannelQuant, CommonUintActQuant
from brevitas.nn.quant_layer import WeightQuantType

IO_DATA_BIT_WIDTH = 8
IO_ACC_BIT_WIDTH = 32


class FastESPCN(FloatESPCN):
    def __init__(self, upscale_factor: int = 2, channels: int = 3):
        super(FastESPCN, self).__init__(upscale_factor=upscale_factor, num_channels=channels)

        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.pixel_shuffle = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )


class FastQuantESPCN(QuantESPCN):
    def __init__(self,
                 upscale_factor: int = 2,
                 num_channels: int = 3,
                 weight_bit_width: int = 4,
                 act_bit_width: int = 4,
                 acc_bit_width: int = 32,
                 weight_quant: WeightQuantType = CommonIntWeightPerChannelQuant
                 ):
        super(FastQuantESPCN, self).__init__(
            upscale_factor=upscale_factor,
            num_channels=num_channels,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            acc_bit_width=acc_bit_width,
            weight_quant=weight_quant
        )

        self.conv4 = qnn.QuantConv2d(
            in_channels=32,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            input_bit_width=act_bit_width,
            input_quant=CommonUintActQuant,
            weight_bit_width=IO_DATA_BIT_WIDTH,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_scaling_per_output_channel=False
        )

        self.pixel_shuffle = qnn.QuantConvTranspose2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=1,
            weight_quant=weight_quant,
            input_quant=None,
            weight_bit_width=IO_DATA_BIT_WIDTH,
            act_bit_width=act_bit_width
        )
