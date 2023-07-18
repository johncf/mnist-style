import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, TypeAlias

import numpy as np
import torch
from scipy import stats
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from mnist_style.persistence import save_models


@dataclass(slots=True)
class TrainMetrics:
    mean_autoenc_loss: float = 0.
    mean_classif_loss: float = 0.
    mean_gener_loss: float = 0.
    mean_discr_fake_loss: float = 0.
    mean_discr_real_loss: float = 0.


@dataclass(slots=True)
class TestMetrics:
    mean_autoenc_loss: float = 0.
    mean_classif_loss: float = 0.
    median_feat_distrib_error: float = 0.


@dataclass(slots=True)
class ModelOptHelper:
    model: nn.Module
    opt: optim.Optimizer

    def __init__(self, model: nn.Module, *, lr: float, optim_cls=optim.AdamW):
        self.model = model
        self.opt = optim_cls(model.parameters(), lr=lr)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode=True):
        return self.model.train(mode)

    def eval(self):
        return self.model.eval()


LossFunction: TypeAlias = Callable[[Tensor, Tensor], Tensor]


@dataclass(slots=True, kw_only=True)
class SimpleTrainer:
    autoencoder: ModelOptHelper
    autoenc_loss_func: LossFunction
    classif_loss_func: LossFunction
    latent_norm_scale: float = 2.

    def train_one_epoch(self, dataloader: DataLoader) -> TrainMetrics:
        cumulative_ae_loss = 0.0
        cumulative_classif_loss = 0.0
        num_samples = 0

        self.autoencoder.train()
        for batch, label in dataloader:
            batch_size = batch.shape[0]
            class_logits, extra_feats, decoded_batch = self.autoencoder(batch)

            ae_loss = self.autoenc_loss_func(decoded_batch, batch)
            classif_loss = self.classif_loss_func(class_logits, label)
            self.autoencoder_optimize(ae_loss + classif_loss)

            cumulative_ae_loss += ae_loss.item() * batch_size
            cumulative_classif_loss += classif_loss.item() * batch_size
            num_samples += batch_size

        return TrainMetrics(
            mean_autoenc_loss=cumulative_ae_loss / num_samples,
            mean_classif_loss=cumulative_classif_loss / num_samples,
        )

    def autoencoder_optimize(self, loss: Tensor) -> None:
        self.autoencoder.opt.zero_grad()
        loss.backward()
        self.autoencoder.opt.step()

    @torch.inference_mode()
    def test_one_epoch(self, dataloader: DataLoader) -> TestMetrics:
        cumulative_ae_loss = 0.0
        cumulative_classif_loss = 0.0
        latent_code_batches = []
        num_samples = 0

        self.autoencoder.eval()
        for batch, label in dataloader:
            batch_size = batch.shape[0]
            class_logits, extra_feats, decoded_batch = self.autoencoder(batch)
            ae_loss = self.autoenc_loss_func(decoded_batch, batch)
            classif_loss = self.classif_loss_func(class_logits, label)

            cumulative_ae_loss += ae_loss.item() * batch_size
            cumulative_classif_loss += classif_loss.item() * batch_size
            latent_code_batches.append(extra_feats.detach().numpy())
            num_samples += batch_size

        latent_codes = np.concatenate(latent_code_batches)
        feat_wise_fit_errors = [
            distribution_fit_error(latent_codes[:, i], norm_scale=self.latent_norm_scale)
            for i in range(latent_codes.shape[1])
        ]
        return TestMetrics(
            mean_autoenc_loss=cumulative_ae_loss / num_samples,
            mean_classif_loss=cumulative_classif_loss / num_samples,
            median_feat_distrib_error=float(np.median(feat_wise_fit_errors)),
        )

    def save_models(self, ckpt_dir) -> None:
        model_dict = {
            "encoder": self.autoencoder.model.encoder,
            "decoder": self.autoencoder.model.decoder,
        }
        save_models(model_dict, ckpt_dir)


@dataclass(slots=True, kw_only=True)
class AdversarialTrainer(SimpleTrainer):
    discriminator: ModelOptHelper
    advers_loss_func: LossFunction

    def train_one_epoch(self, dataloader: DataLoader, gen_loss_factor=0.1) -> TrainMetrics:
        cumulative_ae_loss = 0.0
        cumulative_classif_loss = 0.0
        cumulative_gen_loss = 0.0
        cumulative_dis_fake_loss = 0.0
        cumulative_dis_real_loss = 0.0
        num_samples = 0

        self.autoencoder.train()
        self.discriminator.train()
        for batch, label in dataloader:
            batch_size = batch.shape[0]
            class_logits, style_code, decoded_batch = self.autoencoder(batch)

            fake_input = style_code.detach()  # detach from generator's graph
            fake_output = self.discriminator(fake_input)
            dis_fake_loss = self.advers_loss_func(fake_output, torch.ones_like(fake_output))
            real_input = torch.randn_like(fake_input) * self.latent_norm_scale
            real_output = self.discriminator(real_input)
            dis_real_loss = self.advers_loss_func(real_output, torch.zeros_like(real_output))
            self.discriminator_optimize(dis_fake_loss + dis_real_loss)

            ae_loss = self.autoenc_loss_func(decoded_batch, batch)
            classif_loss = self.classif_loss_func(class_logits, label)
            fake_output = self.discriminator(style_code)  # recompute without detaching
            gen_loss = self.advers_loss_func(fake_output, torch.zeros_like(fake_output))
            self.autoencoder_optimize(ae_loss + classif_loss + gen_loss * gen_loss_factor)

            cumulative_dis_fake_loss += dis_fake_loss.item() * batch_size
            cumulative_dis_real_loss += dis_real_loss.item() * batch_size
            cumulative_ae_loss += ae_loss.item() * batch_size
            cumulative_classif_loss += classif_loss.item() * batch_size
            cumulative_gen_loss += gen_loss.item() * batch_size
            num_samples += batch_size

        return TrainMetrics(
            mean_autoenc_loss=cumulative_ae_loss / num_samples,
            mean_classif_loss=cumulative_classif_loss / num_samples,
            mean_gener_loss=cumulative_gen_loss / num_samples,
            mean_discr_fake_loss=cumulative_dis_fake_loss / num_samples,
            mean_discr_real_loss=cumulative_dis_real_loss / num_samples,
        )

    def discriminator_optimize(self, loss: Tensor) -> None:
        self.discriminator.opt.zero_grad()
        loss.backward()
        self.discriminator.opt.step()

    def save_models(self, ckpt_dir) -> None:
        model_dict = {
            "encoder": self.autoencoder.model.encoder,
            "decoder": self.autoencoder.model.decoder,
            "discriminator": self.discriminator.model,
        }
        save_models(model_dict, ckpt_dir)


def distribution_fit_error(samples, norm_scale=2) -> float:
    cdf = partial(stats.norm.cdf, loc=0, scale=norm_scale)
    ks_test = stats.ks_1samp(samples, cdf)
    # return ks_test.statistic
    return -math.log10(ks_test.pvalue) if ks_test.pvalue > 0 else math.inf
