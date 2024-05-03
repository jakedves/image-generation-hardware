from brevitas_examples.super_resolution.models import FloatESPCN, QuantESPCN

from neural_networks.FastESPCN import FastESPCN, FastQuantESPCN
from neural_networks.MiniESPCN import MiniESPCN, MiniQuantESPCN


class Networks:
    def __init__(self):
        pass

    @staticmethod
    def get(config):
        if config.name == "ESPCN":
            return Networks.espcn(config.quantised, config.bit_width)

        if config.name == "FESPCN":
            return Networks.fespcn(config.quantised, config.bit_width)

        if config.name == "MESPCN":
            return Networks.mespcn(config.quantised, config.bit_width)

        return None

    @staticmethod
    def espcn(quantised, bit_width):
        """
        Returns a 2x upscaling ESPCN network architecture, provided by
        Colbert, Pappalardo, and Petri-Koenig (2023)

        Reference in dissertation
        """
        if not quantised:
            return FloatESPCN(upscale_factor=2)

        return QuantESPCN(
            upscale_factor=2,
            weight_bit_width=bit_width,
            act_bit_width=bit_width
        )

    @staticmethod
    def fespcn(quantised, bit_width):
        """
        Returns a 2x upscaling FastESPCN network architecture
        """
        if not quantised:
            return FastESPCN(upscale_factor=2)

        return FastQuantESPCN(
            upscale_factor=2,
            weight_bit_width=bit_width,
            act_bit_width=bit_width
        )

    @staticmethod
    def mespcn(quantised, bit_width):
        if not quantised:
            return MiniESPCN(upscale_factor=2)

        return MiniQuantESPCN(
            upscale_factor=2,
            weight_bit_width=bit_width,
            act_bit_width=bit_width
        )
