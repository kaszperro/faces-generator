# 0B7EVK8r0v71pZjFTYXZWM3FlRnM
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from DataDownloader import download_file_from_google_drive


class CelebaDataset(Dataset):
    @staticmethod
    def download_celeba_dataset(dataset_folder):
        zip_file = os.path.join(dataset_folder, 'celeba.zip')

        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            download_file_from_google_drive(
                '0B7EVK8r0v71pZjFTYXZWM3FlRnM', zip_file
            )

            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(dataset_folder)

    def __init__(self, dataset_folder: str, download=True, transform=None):
        self.transform = transform
        self.dataset_folder = dataset_folder
        if download:
            self.download_celeba_dataset(self.dataset_folder)
        self.images_folder = os.path.join(self.dataset_folder, 'img_align_celeba')

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

        if self.transform is not None:
            return self.transform(img)

        return img


def display_random_images(dataset_path='./dataset'):
    image_size = 64

    dataloader = DataLoader(
        CelebaDataset(dataset_path, transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])),
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
    display_random_images('./dataset')
