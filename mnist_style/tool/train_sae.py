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

from mnist_style.models import Encoder, Decoder
from mnist_style.persistence import load_models, save_models

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description='MNIST Simple Auto-Encoder')
    parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                        help='batch size for training and testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=8, metavar='E',
                        help='number of epochs to train (default: 8)')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='learning rate with adam optimizer (default: 0.0004)')
    parser.add_argument('--feature-size', type=int, default=8, metavar='N',
                        help='dimensions of the latent feature vector (default: 8)')
    parser.add_argument('--ckpt-dir', default='./pt-sae', metavar='ckpt',
                        help='training session directory (default: ./pt-sae) ' +
                             'for storing model parameters and trainer states')
    parser.add_argument('--data-dir', default='./data', metavar='data',
                        help='MNIST data directory (default: ./data) ' +
                             '(gets created and downloaded to, if doesn\'t exist)')
    opt = parser.parse_args()
    torch.manual_seed(1234)

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

    # Define optimizers
    encoder_opt = optim.AdamW(encoder.parameters(), lr=opt.lr)
    decoder_opt = optim.AdamW(decoder.parameters(), lr=opt.lr)

    # Define loss functions
    autoenc_loss_func = nn.L1Loss()

    for epoch in range(opt.epochs):
        print(f"Epoch {epoch+1} training:")
        encoder.train()
        decoder.train()
        mean_ae_loss = train_one_epoch(
            train_dataloader, encoder, decoder, encoder_opt, decoder_opt, autoenc_loss_func)
        print(f"  Average AutoEnc Loss: {mean_ae_loss:.4f}")
        save_models({"encoder": encoder, "decoder": decoder}, opt.ckpt_dir)
        print(f"Epoch {epoch+1} validation:")
        encoder.eval()
        decoder.eval()
        mean_ae_loss = test_one_epoch(test_dataloader, encoder, decoder, autoenc_loss_func)
        print(f"  Average AutoEnc Loss: {mean_ae_loss:.4f}")
    print("Done!")


def train_one_epoch(dataloader: DataLoader, encoder: Encoder, decoder: Decoder,
                    encoder_opt: optim.Optimizer, decoder_opt: optim.Optimizer,
                    ae_loss_func, g_loss_factor: float = 0.1):
    cumulative_ae_loss = 0.0
    num_batches = 0

    for batch, _ in dataloader:
        # batch = batch.to(device)  # TODO
        latent_code = encoder(batch)
        decoded_batch = decoder(latent_code)

        # Update encoder/generator and decoder
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        ae_loss = ae_loss_func(decoded_batch, batch)
        ae_loss.backward(retain_graph=True)
        encoder_opt.step()
        decoder_opt.step()

        cumulative_ae_loss += ae_loss.item()
        num_batches += 1

    mean_ae_loss = cumulative_ae_loss / num_batches
    return mean_ae_loss


@torch.no_grad()
def test_one_epoch(dataloader: DataLoader, encoder: Encoder, decoder: Decoder, ae_loss_func):
    cumulative_ae_loss = 0.0
    num_batches = 0

    for batch, _ in dataloader:
        latent_code = encoder(batch)
        decoded_batch = decoder(latent_code)
        ae_loss = ae_loss_func(decoded_batch, batch)
        cumulative_ae_loss += ae_loss.item()
        num_batches += 1

    return cumulative_ae_loss / num_batches


if __name__ == '__main__':
    main()
