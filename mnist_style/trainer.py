import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import torch
from scipy import stats
from torch import optim
from torch.utils.data import DataLoader

from mnist_style.models import Decoder, Discriminator, Encoder
from mnist_style.persistence import save_models


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
    autoenc_loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    advers_loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    latent_norm_scale: float = 2.

    def train_one_epoch(self, dataloader: DataLoader, gen_loss_factor=0.1) -> AAETrainMetrics:
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
        feat_wise_fit_errors = [
            distribution_fit_error(latent_codes[:, i], norm_scale=self.latent_norm_scale)
            for i in range(latent_codes.shape[1])
        ]
        return AAETestMetrics(
            mean_autoenc_loss=cumulative_ae_loss / num_samples,
            median_feat_distrib_error=float(np.median(feat_wise_fit_errors)),
        )

    def save_models(self, ckpt_dir):
        save_models({
            "encoder": self.encoder,
            "decoder": self.decoder,
            "discriminator": self.discriminator,
        }, ckpt_dir)


@dataclass(slots=True)
class SAETrainer:
    encoder: Encoder
    decoder: Decoder
    encoder_opt: optim.Optimizer
    decoder_opt: optim.Optimizer
    autoenc_loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        cumulative_ae_loss = 0.0
        num_samples = 0

        self.encoder.train()
        self.decoder.train()
        for batch, label in dataloader:
            batch_size = len(label)
            latent_code = self.encoder(batch)
            decoded_batch = self.decoder(latent_code)

            # Update encoder/generator and decoder
            self.encoder_opt.zero_grad()
            self.decoder_opt.zero_grad()
            ae_loss = self.autoenc_loss_func(decoded_batch, batch)
            ae_loss.backward()
            self.encoder_opt.step()
            self.decoder_opt.step()
            cumulative_ae_loss += ae_loss.item() * batch_size

            num_samples += batch_size

        return cumulative_ae_loss / num_samples

    @torch.inference_mode()
    def test_one_epoch(self, dataloader: DataLoader) -> float:
        cumulative_ae_loss = 0.0
        num_samples = 0

        self.encoder.eval()
        self.decoder.eval()
        for batch, label in dataloader:
            batch_size = len(label)
            latent_code = self.encoder(batch)
            decoded_batch = self.decoder(latent_code)
            ae_loss = self.autoenc_loss_func(decoded_batch, batch)
            cumulative_ae_loss += ae_loss.item() * batch_size
            num_samples += batch_size

        return cumulative_ae_loss / num_samples

    def save_models(self, ckpt_dir):
        save_models({
            "encoder": self.encoder,
            "decoder": self.decoder,
        }, ckpt_dir)


def distribution_fit_error(samples, norm_scale=2) -> float:
    cdf = partial(stats.norm.cdf, loc=0, scale=norm_scale)
    ks_test = stats.ks_1samp(samples, cdf)
    # return ks_test.statistic
    return -math.log10(ks_test.pvalue) if ks_test.pvalue > 0 else math.inf
