from brevitas.export import export_qonnx

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.pytorch import ToTensor

from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as runner
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.util.cleanup import cleanup
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.core.datatype import DataType

from PIL import Image

import numpy as np
import torch

import os


def compare(original, processed):
    loaded_image = Image.open("example.png")
    qonnx_input_tensor = np.array(loaded_image).astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] / 255.0

    input_dict = {
        "global_in": qonnx_input_tensor
    }
    output_dict = runner.execute_onnx(original, input_dict)
    produced_qonnx = output_dict[list(output_dict.keys())[0]]

    finn_input_tensor = np.array(loaded_image).astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]

    input_dict = {
        "global_in": finn_input_tensor
    }

    output_dict = runner.execute_onnx(processed, input_dict)
    produced_finn = output_dict[list(output_dict.keys())[0]]

    # verify that the behaviour is still the same
    similar = np.allclose(produced_qonnx, produced_finn)
    print(f"The outputs are{' NOT' if similar else ''} the same")


def infer_types_and_transform(model):
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    return model


def bake_in_uint8_to_fp32(model):
    input_node_name = model.graph.input[0].name
    input_shape = model.get_tensor_shape(input_node_name)
    div_255 = ToTensor()

    temp_file = "preprocessing.onnx"
    export_qonnx(ToTensor(), torch.randn(input_shape), temp_file)
    cleanup(temp_file, out_file=temp_file)

    preprocessing_model = ModelWrapper(temp_file)
    preprocessing_model = preprocessing_model.transform(ConvertQONNXtoFINN())

    model = model.transform(MergeONNXModels(preprocessing_model))
    global_input_name = model.graph.input[0].name
    model.set_tensor_datatype(global_input_name, DataType["UINT8"])

    return model


def qonnx_to_finn(model):
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())

    return model


def main(r_path, folder):
    onnx_model_path = r_path + folder + "model.onnx"

    original_model = ModelWrapper(onnx_model_path)

    processed_model = qonnx_to_finn(original_model)
    processed_model = bake_in_uint8_to_fp32(processed_model)
    processed_model = infer_types_and_transform(processed_model)

    compare(original_model, processed_model)

    saving_directory = f'../{folder}'
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)
        print(f"Directory '{saving_directory}' created successfully")
    else:
        print(f"Directory '{saving_directory}' already exists")

    processed_model.save(f"{saving_directory}model.onnx")

    temp_file = 'preprocessing.onnx'
    if os.path.exists(temp_file):
        os.remove(temp_file)



if __name__ == '__main__':
    resources_path = 'resources/'

    model_names = [
        'int8-ESPCN',
        'int6-ESPCN',
        'int4-ESPCN',
        'int2-ESPCN',

        'int8-MESPCN',
        'int6-MESPCN',
        'int4-MESPCN',
        'int2-MESPCN',

        'int8-FESPCN',
        'int6-FESPCN',
        'int4-FESPCN',
        'int2-FESPCN',
    ]

    for model_name in ['int8-ESPCN']:
        main(resources_path, model_name + '/')
