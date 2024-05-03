"""
The source code for both of these functions were provided
by Brando_Miranda (2020) on the PyTorch forums :
https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/25
(August 2020), Accessed April 11th 2024
"""


def total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size(model) -> float:
    """
    This function is from Piotr Bialecki (2021)
    August 2021 from the PyTorch forum post: https://discuss.pytorch.org/t/finding-model-size/130275
    """
    param_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb
