#!/bin/env python3

import argparse
import logging
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from mnist_style.models import Encoder, Decoder, Discriminator
from mnist_style.persistence import load_models, save_models


logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description='MNIST Adverserial Auto-Encoder')
    parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                        help='batch size for training and testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=12, metavar='E',
                        help='number of epochs to train (default: 12)')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='learning rate with adam optimizer (default: 0.0004)')
    parser.add_argument('--feature-size', type=int, default=8, metavar='N',
                        help='dimensions of the latent feature vector (default: 8)')
    parser.add_argument('--ckpt-dir', default='./pt-aae', metavar='ckpt',
                        help='training session directory (default: ./pt-aae) ' +
                             'for storing model parameters and trainer states')
    parser.add_argument('--data-dir', default='./data', metavar='data',
                        help='MNIST data directory (default: ./data) ' +
                             '(gets created and downloaded to, if doesn\'t exist)')
    opt = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root=opt.data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=opt.data_dir, train=False, download=False, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=400, shuffle=False)

    # Create model instances
    latent_dim = opt.feature_size
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    discriminator = Discriminator(latent_dim)

    # Define optimizers
    encoder_opt = optim.AdamW(encoder.parameters(), lr=opt.lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=opt.lr)
    discriminator_opt = optim.Adam(discriminator.parameters(), lr=opt.lr)

    # Define loss functions
    autoenc_loss_func = nn.L1Loss()
    adversarial_loss_func = nn.BCELoss()

    encoder.train()
    decoder.train()
    discriminator.train()
    for epoch in range(opt.epochs):
        g_loss_factor = 0.05 * epoch / (opt.epochs - 2)
        print(f"Epoch {epoch+1} training:")
        mean_ae_loss, mean_g_loss, mean_d_loss = train_one_epoch(
            train_dataloader, encoder, decoder, discriminator,
            encoder_opt, decoder_opt, discriminator_opt,
            autoenc_loss_func, adversarial_loss_func, g_loss_factor)
        true_mean_g_loss = math.nan if g_loss_factor == 0.0 else mean_g_loss/g_loss_factor
        print(f"  Average AutoEnc Loss: {mean_ae_loss:.4f}")
        print(f"  Average Generator Loss: {mean_g_loss:.4f} ({true_mean_g_loss:.4f} * {g_loss_factor:.3f})")
        print(f"  Average Discriminator Loss: {mean_d_loss:.4f}")
        save_models({
            "encoder": encoder, "decoder": decoder, "discriminator": discriminator
        }, opt.ckpt_dir)
        # print(f"Epoch {epoch+1} validation:")
        # TODO mean_ae_loss, mean_enc_error = test_one_epoch(test_dataloader, encoder, decoder)
    print("Done!")


def train_one_epoch(dataloader: DataLoader, encoder: Encoder, decoder: Decoder, discriminator: Discriminator,
                    encoder_opt: optim.Optimizer, decoder_opt: optim.Optimizer, discriminator_opt: optim.Optimizer,
                    ae_loss_func, adv_loss_func, g_loss_factor: float = 0.1):
    cumulative_ae_loss = 0.0
    cumulative_d_loss = 0.0
    cumulative_g_loss = 0.0
    num_batches = 0

    for batch, _ in dataloader:
        # batch = batch.to(device)  # TODO
        latent_code = encoder(batch)
        decoded_batch = decoder(latent_code)

        # Update discriminator
        discriminator_opt.zero_grad()
        fake_output = discriminator(latent_code)
        fake_d_loss = adv_loss_func(fake_output, torch.ones_like(fake_output))
        fake_d_loss.backward(retain_graph=True)
        real_output = discriminator(torch.randn_like(latent_code) * 2)  # .to(device)
        real_d_loss = adv_loss_func(real_output, torch.zeros_like(real_output))
        real_d_loss.backward()
        discriminator_opt.step()
        cumulative_d_loss += fake_d_loss.item() + real_d_loss.item()

        # Update encoder/generator and decoder
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        ae_loss = ae_loss_func(decoded_batch, batch)
        ae_loss.backward(retain_graph=True)
        fake_output = discriminator(latent_code)  # recompute, as we updated discriminator above
        fake_g_loss = adv_loss_func(fake_output, torch.zeros_like(fake_output)) * g_loss_factor
        fake_g_loss.backward()
        encoder_opt.step()
        decoder_opt.step()

        cumulative_ae_loss += ae_loss.item()
        cumulative_g_loss += fake_g_loss.item()
        num_batches += 1

    mean_ae_loss = cumulative_ae_loss / num_batches
    mean_g_loss = cumulative_g_loss / num_batches
    mean_d_loss = cumulative_d_loss / num_batches
    return mean_ae_loss, mean_g_loss, mean_d_loss


def test_one_epoch(dataloader: DataLoader, encoder: Encoder, decoder: Decoder, latent_norm_scale: int = 2):
    # TODO return avg_autoencoder_loss, encoder_distribution_errors?
    pass


if __name__ == '__main__':
    main()
