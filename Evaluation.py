import matplotlib.pyplot as plt
import numpy as np
import torch

from Models import get_generator_from_file


def gen_random_image(model_path):
    generator = get_generator_from_file(model_path)
    noise = torch.randn(1, 100, 1, 1)

    out_image = generator(noise).detach().cpu()[0]

    plt.imshow(np.transpose(out_image, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    generator_path = './trained/generator.pth'

    gen_random_image(generator_path)
