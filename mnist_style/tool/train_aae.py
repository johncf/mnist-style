#!/bin/env python3

import argparse
import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from mnist_style.models import ClassifyingAutoEncoder, Discriminator
from mnist_style.trainer import AdversarialTrainer, ModelOptHelper

from .common import cli_parser_add_arguments

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description='MNIST Adverserial Auto-Encoder')
    cli_parser_add_arguments(
        parser, batch_size=64, epochs=12, lr=4e-4, feat_size=4, ckpt_dir='./pt-aae')
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
    autoencoder = ModelOptHelper(ClassifyingAutoEncoder(10, opt.feat_size), lr=opt.lr)
    discriminator = ModelOptHelper(Discriminator(10 + opt.feat_size), lr=opt.lr)

    trainer = AdversarialTrainer(
        autoencoder=autoencoder,
        discriminator=discriminator,
        # Define loss functions
        autoenc_loss_func=nn.L1Loss(),
        classif_loss_func=nn.CrossEntropyLoss(),
        advers_loss_func=nn.BCEWithLogitsLoss(),
    )

    for epoch in range(opt.epochs):
        gen_loss_factor = 0.2 * epoch / max(1, opt.epochs - 2)
        print(f"Epoch {epoch+1} training:", flush=True)
        t_start = time.time()
        train_metrics = trainer.train_one_epoch(train_dataloader, gen_loss_factor)
        t_end = time.time()
        print(f"  Mean AutoEncoder Loss: {train_metrics.mean_autoenc_loss:.4f}")
        print(f"  Mean Classification Loss: {train_metrics.mean_classif_loss:.4f}")
        print(f"  Mean Generator Loss: {train_metrics.mean_gener_loss:.4f} * {gen_loss_factor:.3f}")
        print(f"  Mean Discriminator Fake Loss: {train_metrics.mean_discr_fake_loss:.4f}")
        print(f"  Mean Discriminator Real Loss: {train_metrics.mean_discr_real_loss:.4f}")
        trainer.save_models(opt.ckpt_dir)
        print(f"Epoch {epoch+1} validation:", flush=True)
        v_start = time.time()
        test_metrics = trainer.test_one_epoch(test_dataloader)
        v_end = time.time()
        print(f"  Mean AutoEncoder Loss: {test_metrics.mean_autoenc_loss:.4f}")
        print(f"  Mean Classification Loss: {test_metrics.mean_classif_loss:.4f}")
        print(f"  Median Encoded Distribution Error: {test_metrics.median_feat_distrib_error:.4f}")
        print(f"[time] training: {t_end - t_start:.1f}s, validation: {v_end - v_start:.1f}s")
    print("Done!")


if __name__ == '__main__':
    main()
