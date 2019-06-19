import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, num_input=100, num_futures=64, color_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_input, num_futures * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_futures * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_futures * 8, num_futures * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_futures * 4, num_futures * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_futures * 2, num_futures, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_futures, color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        #
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, num_futures=64, color_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(color_channels, num_futures, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_futures, num_futures * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_futures * 2, num_futures * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_futures * 4, num_futures * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_futures * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_futures * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def stat_model_parameters(model):
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def stat_generator_parameters():
    stat_model_parameters(Generator())


def stat_discriminator_parameters():
    stat_model_parameters(Discriminator())


def get_generator_from_file(file_path):
    num_input = 100
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gen = Generator(num_input=num_input, ).to(device)
    gen.load_state_dict(torch.load(file_path, map_location='cpu'))
    return gen.eval()


if __name__ == '__main__':
    stat_generator_parameters()
    stat_discriminator_parameters()
