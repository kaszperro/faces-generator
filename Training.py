import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils, transforms

from FacesDataset import CelebaDataset
from Models import Generator, Discriminator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_model(model, device, num_gpu):
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (num_gpu > 1):
        model = nn.DataParallel(model, list(range(num_gpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    model.apply(weights_init)

    # Print the model
    print(model)

    return model


def get_generator(device, num_gpu, num_input):
    model = Generator(num_input=num_input).to(device)

    return init_model(model, device, num_gpu)


def get_discriminator(device, num_gpu):
    model = Discriminator().to(device)

    return init_model(model, device, num_gpu)


def save_images(generator, noise, directory, file_name, title):
    with torch.no_grad():
        fake = generator(noise).detach().cpu()

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(utils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(os.path.join(directory, file_name))


def train(save_images_dir):
    # parameters
    num_gpu = 1
    num_input = 100
    real_label = 1
    fake_label = 0
    lr = 0.0002
    beta1 = 0.5
    num_epochs = 5
    image_size = 64
    batch_size = 64
    # -----

    device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

    gen = get_generator(device, num_gpu, num_input)
    disc = get_discriminator(device, num_gpu)

    loss = nn.BCELoss()

    testing_noise = torch.randn(64, num_input, 1, 1, device=device)

    disc_optim = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))
    gen_optim = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

    data_loader = DataLoader(
        CelebaDataset('./dataset', transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    iteration = 0

    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(data_loader):
            disc.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = disc(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = loss(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, num_input, 1, 1, device=device)
            # Generate fake image batch with G
            fake = gen(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = disc(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = loss(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            disc_optim.step()

            gen.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = disc(fake).view(-1)
            # Calculate G's loss based on this output
            errG = loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            gen_optim.step()

            if i % 100 == 0:
                save_images(gen, testing_noise, save_images_dir,
                            'iteration_{}.png'.format(iteration),
                            "Generated images, iteration: {}".format(iteration))

            iteration += 1

    torch.save(disc.state_dict(), 'discriminator.pth')
    torch.save(gen.state_dict(), 'generator.pth')


if __name__ == '__main__':
    train('./generated/')
