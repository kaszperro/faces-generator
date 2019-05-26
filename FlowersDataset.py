import os
import shutil
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from DataDownloader import download_file_from_url


class FlowersDataset(Dataset):
    @staticmethod
    def download_dataset(save_path, extract_directory):
        url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        if not os.path.exists(save_path):
            download_file_from_url(url, save_path)

        if not os.path.exists(extract_directory):

            tr = tarfile.open(save_path)
            tr.extractall(extract_directory)
            tr.close()

            files = os.listdir(os.path.join(extract_directory, 'jpg'))

            for f in files:
                shutil.move(os.path.join(extract_directory, 'jpg', f), extract_directory)

            shutil.rmtree(os.path.join(extract_directory, 'jpg'))

    def __init__(self, image_size=64):
        self.images_folder = './dataset/flowers/'

        self.download_dataset('./dataset/flowers.tgz', self.images_folder)
        self.image_size = image_size

        self.all_images_paths = []

        for (_, _, filenames) in os.walk(self.images_folder):
            self.all_images_paths.extend(filenames)

    def __len__(self):
        return len(self.all_images_paths)

    @staticmethod
    def load_image(image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.all_images_paths[idx])

        img = self.load_image(image_path)

        my_trans = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        return my_trans(img)


def display_random_images():
    dataloader = DataLoader(
        FlowersDataset(),
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    real_batch = next(iter(dataloader))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[:64], padding=2, normalize=True), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    display_random_images()
