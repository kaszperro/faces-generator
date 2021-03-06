import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils

import FacesDataset
from FlowersDataset import FlowersDataset
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


def train(dataset, save_path, num_epochs=5):
    # main train function
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # parameters
    num_gpu = 1
    num_input = 100
    real_label = 1
    fake_label = 0
    lr = 0.0002
    beta1 = 0.5

    image_size = 64
    batch_size = 64
    # -----

    device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

    gen = get_generator(device, num_gpu, num_input)
    disc = get_discriminator(device, num_gpu)

    loss = nn.BCELoss()

    testing_noise = torch.randn(64, num_input, 1, 1, device=device)

    torch.save(testing_noise, os.path.join(save_path, 'noise.pth'))

    disc_optim = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))
    gen_optim = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    iteration = 0

    g_losses = []
    d_losses = []

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

            if iteration % 100 == 0:
                # saves generated images after each 100 iterations, we can see progress
                save_images(gen, testing_noise, os.path.join(save_path, 'generated'),
                            'iteration_{}.png'.format(iteration),
                            "Generated images, iteration: {}".format(iteration))

                g_losses.append(errG.item())
                d_losses.append(errD.item())

            iteration += 1
    # after we finish trainings, we also save
    # generator and discriminator models, to restore them later
    torch.save(disc.state_dict(), os.path.join(save_path, 'discriminator.pth'))
    torch.save(gen.state_dict(), os.path.join(save_path, 'generator.pth'))

    with open('generator_loss', 'wb') as fp:
        pickle.dump(g_losses, fp)

    with open('discriminator_loss', 'wb') as fp:
        pickle.dump(d_losses, fp)


def train_flowers():
    # we use same function for faces and flowers
    train(FlowersDataset(), os.path.join('trained', 'flowers'), 40)


def train_faces():
    # we use same function for faces and flowers
    train(FacesDataset.CelebaDataset('./dataset/'), os.path.join('trained', 'faces'), 10)


if __name__ == '__main__':
    train_flowers()
    # train_faces()
