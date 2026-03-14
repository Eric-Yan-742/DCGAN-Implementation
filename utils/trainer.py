import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def Trainer(Discriminator,
            Generator,
            dataloader,
            num_epochs,
            nz=100,
            lr=0.0002,
            beta1=0.5):
    # Use binary cross entropy as the loss function
    loss_function = nn.BCELoss()

    # fixed noise for visualizing progress
    visualization_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    discriminator_optimizer = optim.Adam(Discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    generator_optimizer = optim.Adam(Generator.parameters(), lr=lr, betas=(beta1, 0.999))

    real_target_label = 1.
    fake_target_label = 0.

    trained_generator, trained_discriminator, generator_losses, discriminator_losses, generated_image_grids = Train(Discriminator,
                               Generator,
                               loss_function,
                               discriminator_optimizer,
                               generator_optimizer,
                               num_epochs,
                               nz,
                               dataloader,
                               visualization_noise,
                               real_target_label,
                               fake_target_label)

    return trained_generator, trained_discriminator, generator_losses, discriminator_losses, generated_image_grids

# Training loop
def Train(Discriminator,
          Generator,
          criterion,
          optimizerD,
          optimizerG,
          num_epochs,
          nz,
          dataloader,
          fixed_noise,
          real_label,
          fake_label,
          ):
    generated_image_grids = []
    generator_losses = []
    discriminator_losses = []

    # Keep track of total iterations
    iteration_count = 0
    total_training_steps = num_epochs * len(dataloader)
    snapshot_interval = total_training_steps // 10

    for epoch_index in range(num_epochs):
        for batch_index, (real_images, batch_targets) in enumerate(dataloader):

            ## Train discrimiantor with a real batch
            Discriminator.zero_grad()
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            target_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            discriminator_predictions = Discriminator(real_images).view(-1)
            discriminator_real_loss = criterion(discriminator_predictions, target_labels)
            # backprop with the real batch
            discriminator_real_loss.backward()

            ## Train discriminator with a fake batch
            latent_noise = torch.randn(batch_size, nz, 1, 1, device=device)
            generated_images = Generator(latent_noise)
            target_labels.fill_(fake_label)
            discriminator_predictions = Discriminator(generated_images.detach()).view(-1)
            discriminator_fake_loss = criterion(discriminator_predictions, target_labels)
            # back prop with the fake batch, accumulate gradient
            discriminator_fake_loss.backward()
            discriminator_total_loss = discriminator_real_loss + discriminator_fake_loss
            # Update discriminator
            optimizerD.step()

            # Train generator with the fake batch
            Generator.zero_grad()
            target_labels.fill_(real_label)
            discriminator_predictions = Discriminator(generated_images).view(-1)
            generator_loss = criterion(discriminator_predictions, target_labels)
            # backprop D and G with the fake batch
            generator_loss.backward()
            # Update generator
            optimizerG.step()


            # Save Losses for plotting later
            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_total_loss.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iteration_count % snapshot_interval == 0) or ((epoch_index == num_epochs-1) and (batch_index == len(dataloader)-1)):
                with torch.no_grad():
                    preview_images = Generator(fixed_noise).detach().cpu()
                generated_image_grids.append(vutils.make_grid(preview_images, padding=2, normalize=True))

            iteration_count += 1
    return Generator, Discriminator, generator_losses, discriminator_losses, generated_image_grids


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# visualize a batch from dataloader
def visualize(dataloader):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

# visualize a batch (64) from RGB
def show_batch(images, start_idx=0, nrow=8, figsize=(10,10)):
    n = len(images)
    ncol = nrow
    nrow = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue

        img = images[i].permute(1,2,0)
        img = (img + 1) / 2
        ax.imshow(img)   # works with Normalize((-0.5),(0.5))
        ax.set_title(str(start_idx + i), fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"indices: {start_idx} to {start_idx + n - 1}")

def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feat = model.extract_features(imgs).cpu().numpy()
            features.append(feat)
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)
