#!/bin/env python3

import argparse
import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from mnist_style.models import Decoder, Encoder
from mnist_style.trainer import SAETrainer

from .common import cli_parser_add_arguments

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description='MNIST Simple Auto-Encoder')
    cli_parser_add_arguments(
        parser, batch_size=64, epochs=10, lr=4e-4, feat_size=8, ckpt_dir='./pt-sae')
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

    # Create model instances
    encoder = Encoder(opt.feat_size)
    decoder = Decoder(opt.feat_size)

    trainer = SAETrainer(
        encoder=encoder,
        decoder=decoder,
        # Define optimizers
        encoder_opt=optim.AdamW(encoder.parameters(), lr=opt.lr),
        decoder_opt=optim.AdamW(decoder.parameters(), lr=opt.lr),
        # Define loss functions
        autoenc_loss_func=nn.L1Loss(),
    )

    for epoch in range(opt.epochs):
        print(f"Epoch {epoch+1} training:")
        encoder.train()
        decoder.train()
        mean_ae_loss = trainer.train_one_epoch(train_dataloader)
        print(f"  Mean AutoEncoder Loss: {mean_ae_loss:.4f}")
        trainer.save_models(opt.ckpt_dir)
        print(f"Epoch {epoch+1} validation:")
        encoder.eval()
        decoder.eval()
        mean_ae_loss = trainer.test_one_epoch(test_dataloader)
        print(f"  Mean AutoEncoder Loss: {mean_ae_loss:.4f}")
    print("Done!")


if __name__ == '__main__':
    main()
