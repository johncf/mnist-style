#!/bin/env python3

import argparse
import logging
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy import stats
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
    discriminator = Discriminator(latent_dim)

    # Define optimizers
    encoder_opt = optim.AdamW(encoder.parameters(), lr=opt.lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=opt.lr)
    discriminator_opt = optim.Adam(discriminator.parameters(), lr=opt.lr)

    # Define loss functions
    autoenc_loss_func = nn.L1Loss()
    adversarial_loss_func = nn.BCELoss()

    discriminator.train()
    for epoch in range(opt.epochs):
        gen_loss_factor = 0.05 * epoch / (opt.epochs - 2)
        print(f"Epoch {epoch+1} training:")
        encoder.train()
        decoder.train()
        mean_ae_loss, mean_dis_fake_loss, mean_dis_real_loss = train_one_epoch(
            train_dataloader, encoder, decoder, discriminator,
            encoder_opt, decoder_opt, discriminator_opt,
            autoenc_loss_func, adversarial_loss_func, gen_loss_factor)
        print(f"  Average AutoEnc Loss: {mean_ae_loss:.4f} (gen factor: {gen_loss_factor:.3f})")
        print(f"  Average Discriminator Fake Loss: {mean_dis_fake_loss:.4f}")
        print(f"  Average Discriminator Real Loss: {mean_dis_real_loss:.4f}", flush=True)
        save_models({
            "encoder": encoder, "decoder": decoder, "discriminator": discriminator
        }, opt.ckpt_dir)
        print(f"Epoch {epoch+1} validation:")
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            mean_ae_loss, median_enc_error = test_one_epoch(test_dataloader, encoder, decoder, autoenc_loss_func)
        print(f"  Average AutoEnc Loss: {mean_ae_loss:.4f}")
        print(f"  Median Encoded Distribution Error: {median_enc_error:.4f}", flush=True)
    print("Done!")


def train_one_epoch(dataloader: DataLoader, encoder: Encoder, decoder: Decoder, discriminator: Discriminator,
                    encoder_opt: optim.Optimizer, decoder_opt: optim.Optimizer, discriminator_opt: optim.Optimizer,
                    ae_loss_func, adv_loss_func, gen_loss_factor: float = 0.1, latent_norm_scale: float = 2.):
    cumulative_ae_loss = 0.0
    cumulative_dis_fake_loss = 0.0
    cumulative_dis_real_loss = 0.0
    num_batches = 0

    for batch, _ in dataloader:
        # batch = batch.to(device)  # TODO
        latent_code = encoder(batch)
        decoded_batch = decoder(latent_code)

        # Update discriminator
        discriminator_opt.zero_grad()
        fake_output = discriminator(latent_code)
        dis_fake_loss = adv_loss_func(fake_output, torch.ones_like(fake_output))
        dis_fake_loss.backward(retain_graph=True)
        real_output = discriminator(torch.randn_like(latent_code) * latent_norm_scale)
        dis_real_loss = adv_loss_func(real_output, torch.zeros_like(real_output))
        dis_real_loss.backward()
        discriminator_opt.step()
        cumulative_dis_fake_loss += dis_fake_loss.item()
        cumulative_dis_real_loss += dis_real_loss.item()

        # Update encoder/generator and decoder
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        ae_loss = ae_loss_func(decoded_batch, batch)
        ae_loss.backward(retain_graph=True)
        fake_output = discriminator(latent_code)  # recompute, as we updated discriminator above
        gen_loss = adv_loss_func(fake_output, torch.zeros_like(fake_output)) * gen_loss_factor
        gen_loss.backward()
        encoder_opt.step()
        decoder_opt.step()

        cumulative_ae_loss += ae_loss.item()
        num_batches += 1

    mean_ae_loss = cumulative_ae_loss / num_batches
    mean_dis_fake_loss = cumulative_dis_fake_loss / num_batches
    mean_dis_real_loss = cumulative_dis_real_loss / num_batches
    return mean_ae_loss, mean_dis_fake_loss, mean_dis_real_loss


def test_one_epoch(dataloader: DataLoader, encoder: Encoder, decoder: Decoder, ae_loss_func, latent_norm_scale: float = 2.):
    cumulative_ae_loss = 0.0
    latent_code_batches = []
    num_batches = 0

    for batch, _ in dataloader:
        latent_code = encoder(batch)
        decoded_batch = decoder(latent_code)
        ae_loss = ae_loss_func(decoded_batch, batch)
        cumulative_ae_loss += ae_loss.item()
        latent_code_batches.append(latent_code.detach().numpy())
        num_batches += 1

    mean_ae_loss = cumulative_ae_loss / num_batches
    latent_codes = np.concatenate(latent_code_batches)
    feat_wise_logps = [distribution_fit_error(latent_codes[:, i], norm_scale=latent_norm_scale)
                       for i in range(latent_codes.shape[1])]
    median_feat_enc_error = np.median(feat_wise_logps)
    return mean_ae_loss, median_feat_enc_error


def distribution_fit_error(samples, norm_scale=2) -> float:
    cdf = partial(stats.norm.cdf, loc=0, scale=norm_scale)
    ks_test = stats.ks_1samp(samples, cdf)
    # return ks_test.statistic
    return -math.log10(ks_test.pvalue) if ks_test.pvalue > 0 else math.inf


if __name__ == '__main__':
    main()
