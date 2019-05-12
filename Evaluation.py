import matplotlib.pyplot as plt
import numpy as np
import torch

from Models import get_generator_from_file


def generate_image_from_vector(input_vector, model_path='./trained/generator.pth'):
    """

    :param model_path: path to model
    :param input_vector: vector of 100 numbers
    :return: image as numpy array of shape (3, 64, 64)
    """
    generator = get_generator_from_file(model_path)

    noise = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    out_image = generator(noise).detach().cpu()[0]

    return np.transpose(out_image, (1, 2, 0))


def display_random_image(model_path='./trained/generator.pth'):
    noise = np.random.rand(100)

    out_image = generate_image_from_vector(noise, model_path)

    plt.imshow(out_image)
    plt.show()


if __name__ == '__main__':
    display_random_image()
