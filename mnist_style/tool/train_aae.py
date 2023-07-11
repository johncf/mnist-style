#!/bin/env python3

import argparse
import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

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
                        help='learning rate with adam optimizer (default: 4e-4)')
    parser.add_argument('--feature-size', type=int, default=8, metavar='N',
                        help='dimensions of the latent feature vector (default: 8)')
    parser.add_argument('--ckpt-dir', default='./pt-aae', metavar='ckpt',
                        help='training session directory (default: ./pt-aae) ' +
                             'for storing model parameters and trainer states')
    parser.add_argument('--data-dir', default='./data', metavar='data',
                        help='MNIST data directory (default: ./data) ' +
                             '(gets created and downloaded to, if doesn\'t exist)')
    opt = parser.parse_args()
    torch.manual_seed(1235)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root=opt.data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=opt.data_dir, train=False, download=False, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4 * opt.batch_size, shuffle=False)

    # Create model instances
    latent_dim = opt.feature_size
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    discriminator = Discriminator(latent_dim)

    trainer = AAETrainer(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        # Define optimizers
        encoder_opt=optim.AdamW(encoder.parameters(), lr=opt.lr),
        decoder_opt=optim.AdamW(decoder.parameters(), lr=opt.lr),
        discriminator_opt=optim.AdamW(discriminator.parameters(), lr=opt.lr),
        # Define loss functions
        autoenc_loss_func=nn.L1Loss(),
        advers_loss_func=nn.BCEWithLogitsLoss(),
    )

    for epoch in range(opt.epochs):
        gen_loss_factor = 0.1 * max(0, epoch - 1) / (opt.epochs - 2)
        print(f"Epoch {epoch+1} training:", flush=True)
        train_metrics = trainer.train_one_epoch(train_dataloader, gen_loss_factor)
        print(f"  Mean AutoEncoder Loss: {train_metrics.mean_autoenc_loss:.4f}")
        print(f"  Mean Generator Loss: {train_metrics.mean_gener_loss:.4f} * {gen_loss_factor:.3f}")
        print(f"  Mean Discriminator Fake Loss: {train_metrics.mean_discr_fake_loss:.4f}")
        print(f"  Mean Discriminator Real Loss: {train_metrics.mean_discr_real_loss:.4f}")
        trainer.save_models(opt.ckpt_dir)
        print(f"Epoch {epoch+1} validation:", flush=True)
        test_metrics = trainer.test_one_epoch(test_dataloader)
        print(f"  Mean AutoEncoder Loss: {test_metrics.mean_autoenc_loss:.4f}")
        print(f"  Median Encoded Distribution Error: {test_metrics.median_feat_distrib_error:.4f}")
    print("Done!")


@dataclass(slots=True)
class AAETrainMetrics:
    mean_autoenc_loss: float = 0.
    mean_gener_loss: float = 0.
    mean_discr_fake_loss: float = 0.
    mean_discr_real_loss: float = 0.


@dataclass(slots=True)
class AAETestMetrics:
    mean_autoenc_loss: float = 0.
    median_feat_distrib_error: float = 0.


@dataclass(slots=True)
class AAETrainer:
    encoder: Encoder
    decoder: Decoder
    discriminator: Discriminator
    encoder_opt: optim.Optimizer
    decoder_opt: optim.Optimizer
    discriminator_opt: optim.Optimizer
    autoenc_loss_func: Callable
    advers_loss_func: Callable
    latent_norm_scale: float = 2.

    def train_one_epoch(self, dataloader: DataLoader, gen_loss_factor: float = 0.1) -> AAETrainMetrics:
        cumulative_ae_loss = 0.0
        cumulative_gen_loss = 0.0
        cumulative_dis_fake_loss = 0.0
        cumulative_dis_real_loss = 0.0
        num_samples = 0

        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        for batch, label in dataloader:
            batch_size = len(label)
            latent_code = self.encoder(batch)
            decoded_batch = self.decoder(latent_code)

            # Update discriminator
            fake_output = self.discriminator(latent_code.detach())
            dis_fake_loss = self.advers_loss_func(fake_output, torch.ones_like(fake_output))
            real_output = self.discriminator(torch.randn_like(latent_code) * self.latent_norm_scale)
            dis_real_loss = self.advers_loss_func(real_output, torch.zeros_like(real_output))
            self.discriminator_opt.zero_grad()
            (dis_fake_loss + dis_real_loss).backward()
            self.discriminator_opt.step()
            cumulative_dis_fake_loss += dis_fake_loss.item() * batch_size
            cumulative_dis_real_loss += dis_real_loss.item() * batch_size

            # Update encoder/generator and decoder
            ae_loss = self.autoenc_loss_func(decoded_batch, batch)
            fake_output = self.discriminator(latent_code)  # recompute without detaching
            gen_loss = self.advers_loss_func(fake_output, torch.zeros_like(fake_output))
            self.encoder_opt.zero_grad()
            self.decoder_opt.zero_grad()
            (ae_loss + gen_loss * gen_loss_factor).backward()
            self.encoder_opt.step()
            self.decoder_opt.step()
            cumulative_gen_loss += gen_loss.item() * batch_size
            cumulative_ae_loss += ae_loss.item() * batch_size

            num_samples += batch_size

        return AAETrainMetrics(
            mean_autoenc_loss=cumulative_ae_loss / num_samples,
            mean_gener_loss=cumulative_gen_loss / num_samples,
            mean_discr_fake_loss=cumulative_dis_fake_loss / num_samples,
            mean_discr_real_loss=cumulative_dis_real_loss / num_samples,
        )

    @torch.inference_mode()
    def test_one_epoch(self, dataloader: DataLoader) -> AAETestMetrics:
        cumulative_ae_loss = 0.0
        latent_code_batches = []
        num_samples = 0

        self.encoder.eval()
        self.decoder.eval()
        for batch, label in dataloader:
            batch_size = len(label)
            latent_code = self.encoder(batch)
            decoded_batch = self.decoder(latent_code)
            ae_loss = self.autoenc_loss_func(decoded_batch, batch)
            cumulative_ae_loss += ae_loss.item() * batch_size
            latent_code_batches.append(latent_code.detach().numpy())
            num_samples += batch_size

        latent_codes = np.concatenate(latent_code_batches)
        feat_wise_fit_errors = [distribution_fit_error(latent_codes[:, i], norm_scale=self.latent_norm_scale)
                                for i in range(latent_codes.shape[1])]
        return AAETestMetrics(
            mean_autoenc_loss=cumulative_ae_loss / num_samples,
            median_feat_distrib_error=np.median(feat_wise_fit_errors),
        )

    def save_models(self, ckpt_dir):
        save_models({
            "encoder": self.encoder,
            "decoder": self.decoder,
            "discriminator": self.discriminator,
        }, ckpt_dir)


def distribution_fit_error(samples, norm_scale=2) -> float:
    cdf = partial(stats.norm.cdf, loc=0, scale=norm_scale)
    ks_test = stats.ks_1samp(samples, cdf)
    # return ks_test.statistic
    return -math.log10(ks_test.pvalue) if ks_test.pvalue > 0 else math.inf


if __name__ == '__main__':
    main()
