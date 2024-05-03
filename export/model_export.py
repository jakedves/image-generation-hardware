import torch
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup


def onnx_export(network, dummy_input_shape, cfg, folder):
    dummy_input = torch.randn(dummy_input_shape)
    model = network.to('cpu')
    model.eval()

    filename = folder + "/model.onnx"

    export_f = export_qonnx if cfg.quantised else torch.onnx.export

    export_f(
        model,
        dummy_input,
        filename,
        verbose=False
    )

    if cfg.quantised:
        qonnx_cleanup(filename, out_file=filename)


def torch_export(network, folder):
    model = network.to('cpu')
    model.eval()

    torch.save(model.state_dict(), folder + "/model.pth")
