# image-generation-hardware

This repository holds the (very unpolished) code for my dissertation.

There are three main files than can be run.

`main.py` will train a neural network regularly, possibly with quantisation (edit the file to adjust)

`main_kd.py` will train a teacher model and numerous student models, each with quantisation

`compile.py` will apply transformations to a QONNX compute graph, converting it into a FINN-ONNX compute graph, and baking in ```UINT8 -> FP32``` image input coversion

The main files can be ran on any machine with the following command, although may require CUDA support.

`python3 main.py`

The compilation file requires the setup of a linux virtual machine, docker, and installation of the Vitis development environment. Futher details can be found in the [FINN documentation](https://finn.readthedocs.io/en/latest/getting_started.html). For generating hardware, the command line was used, with the `dataflow_build_config.json` file.

**TL;DR:**

The main files should run fine, good luck running `compile.py`.
