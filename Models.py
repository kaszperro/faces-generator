import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, num_input=100, num_futures=64, color_channels=3):
        super().__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_input, num_futures * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_futures * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_futures * 8, num_futures * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(num_futures * 4, num_futures * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(num_futures * 2, num_futures, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_futures, color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, num_futures=64, color_channels=3):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(color_channels, num_futures, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_futures, num_futures * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_futures * 2, num_futures * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_futures * 4, num_futures * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_futures * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def get_generator_from_file(file_path):
    num_input = 100
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gen = Generator(num_input=num_input).to(device)
    gen.load_state_dict(torch.load(file_path))
    return gen.eval()
