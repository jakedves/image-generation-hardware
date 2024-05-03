from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

from device import DEVICE


def generate_image(generate, path):
    """
    Given a generating function, attempt image generation
    """
    test_image = Image.open(path)

    # unsqueeze(0) inserts the batch dimension to the 0th index
    test_image = ToTensor()(test_image).unsqueeze(0).to(DEVICE)

    resulting_image = generate(test_image)
    resulting_image = resulting_image.detach().cpu().numpy()[0]
    resulting_image = np.transpose(resulting_image, (1, 2, 0))
    resulting_image = resulting_image * 255.0
    resulting_image = resulting_image.astype(np.uint8)

    return resulting_image
