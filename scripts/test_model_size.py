from brevitas_examples.super_resolution.models import QuantESPCN
from metrics import model_size

for bit_width in [1, 2, 4, 8, 16]:
    model = QuantESPCN(weight_bit_width=bit_width, act_bit_width=bit_width)
    print(model_size(model))
    # 0.2621574401855469
    # 0.2621574401855469
    # 0.2621574401855469
    # 0.2621574401855469
    # 0.2621574401855469
