from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def render(t):
    cols = 5
    figure, axes = plt.subplots(nrows=cols, ncols=cols, figsize=(5, 5))
    axes = axes.flatten()

    for i in range(24):
        img = t[i].numpy().transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = '../datasets/super-resolution/BSD100/img_001_SRF_2_HR.png'

    image = Image.open(path)
    tensor = transforms.ToTensor()(image)

    patch_size = 80

    image_height = tensor.shape[1]
    image_width = tensor.shape[2]

    patch_count = (image_height // patch_size) * (image_width // patch_size)
    print(f"{tensor.shape} - loaded image")

    y = tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    print(f"{y.shape} - unfolded image")

    y = y.reshape(3, patch_count, patch_size, patch_size)
    print(f"{y.shape} - reshaped image")

    y = y.permute(1, 0, 2, 3)
    print(f"{y.shape} - final image batch")

    render(y)



