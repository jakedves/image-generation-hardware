import torch
from neural_networks.FastESPCN import FastESPCN, FastQuantESPCN
from neural_networks.MiniESPCN import MiniESPCN, MiniQuantESPCN
from neural_networks.FastMiniESPCN import FastMiniESPCN, FastMiniQuantESPCN
from brevitas_examples.super_resolution.models import FloatESPCN, QuantESPCN


if __name__ == '__main__':
    tensor = torch.randn(1, 3, 512, 512)

    model = QuantESPCN(upscale_factor=2)
    out = model(tensor)
    print(out.shape)

    model = FloatESPCN(upscale_factor=2)
    out = model(tensor)
    print(out.shape)

    model = FastESPCN()
    out = model(tensor)
    print(out.shape)

    model = FastQuantESPCN()
    out = model(tensor)
    print(out.shape)

    model = MiniESPCN()
    out = model(tensor)
    print(out.shape)

    model = MiniQuantESPCN()
    out = model(tensor)
    print(out.shape)

    model = FastMiniESPCN()
    out = model(tensor)
    print(out.shape)

    model = FastMiniQuantESPCN()
    out = model(tensor)
    print(out.shape)


