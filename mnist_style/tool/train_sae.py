#!/bin/env python3

import argparse
import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from mnist_style.models import ClassifyingAutoEncoder
from mnist_style.trainer import ModelOptHelper, SimpleTrainer

from .common import cli_parser_add_arguments

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description='MNIST Simple Auto-Encoder')
    cli_parser_add_arguments(
        parser, batch_size=64, epochs=10, lr=4e-4, feat_size=4, ckpt_dir='./pt-sae')
    opt = parser.parse_args()
    torch.manual_seed(1234)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root=opt.data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=opt.data_dir, train=False, download=False, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4 * opt.batch_size, shuffle=False)

    # Create model instance + optimizer wrapper
    autoencoder = ModelOptHelper(ClassifyingAutoEncoder(10, opt.feat_size), lr=opt.lr)

    trainer = SimpleTrainer(
        autoencoder=autoencoder,
        # Define loss functions
        autoenc_loss_func=nn.L1Loss(),
        classif_loss_func=nn.CrossEntropyLoss(),
    )

    for epoch in range(opt.epochs):
        print(f"Epoch {epoch+1} training:")
        t_start = time.time()
        train_metrics = trainer.train_one_epoch(train_dataloader)
        t_end = time.time()
        print(f"  Mean AutoEncoder Loss: {train_metrics.mean_autoenc_loss:.4f}")
        print(f"  Mean Classification Loss: {train_metrics.mean_classif_loss:.4f}")
        trainer.save_models(opt.ckpt_dir)
        print(f"Epoch {epoch+1} validation:")
        v_start = time.time()
        test_metrics = trainer.test_one_epoch(test_dataloader)
        v_end = time.time()
        print(f"  Mean AutoEncoder Loss: {test_metrics.mean_autoenc_loss:.4f}")
        print(f"  Mean Classification Loss: {test_metrics.mean_classif_loss:.4f}")
        print(f"[time] training: {t_end - t_start:.1f}s, validation: {v_end - v_start:.1f}s")
    print("Done!")


if __name__ == '__main__':
    main()
