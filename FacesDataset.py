# 0B7EVK8r0v71pZjFTYXZWM3FlRnM
import os
import zipfile

from torch.utils.data import Dataset

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
                zip_ref.extractall(os.path.join(dataset_folder, 'images'))

    def __init__(self, dataset_folder: str, download=True):
        self.dataset_folder = dataset_folder
        if download:
            self.download_celeba_dataset(self.dataset_folder)


if __name__ == '__main__':
    ds = CelebaDataset('./dataset')
