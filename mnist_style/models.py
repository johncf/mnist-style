from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_groups=2, **conv2d_args):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, bias=False, **conv2d_args),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(True),
        )


class Encoder(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self._seq = nn.Sequential(
            ConvBlock(1, 16, kernel_size=5, stride=2, padding=2, norm_groups=4),
            ConvBlock(16, 32, kernel_size=3, stride=1, padding=1, norm_groups=8),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, num_features)
        )

    def forward(self, batch: Tensor) -> Tensor:
        return self._seq(batch)


class Decoder(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.fc = nn.Linear(num_features, 32 * 7 * 7)
        self.conv1 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1, norm_groups=8)
        self.conv2 = ConvBlock(32, 4, kernel_size=3, stride=1, padding=1, norm_groups=2)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, batch: Tensor) -> Tensor:
        x = self.fc(batch)
        x = x.view(-1, 32, 7, 7)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ClassifyingAutoEncoder(nn.Module):
    def __init__(self, num_classes: int, num_features: int):
        super().__init__()
        self.encoder = Encoder(num_classes + num_features)
        self.decoder = Decoder(num_classes + num_features)
        self.num_classes = num_classes

    def forward(
        self, batch: Tensor, class_labels: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        class_logits, extra_feats = self.forward_encoder(batch)
        # we don't want to backpropagate from decoder to encoder through class logits,
        # as we don't want information to flow this way; so use detached logits
        detached_logits = class_logits.detach()
        if class_labels is not None:  # boost true labels during training
            detached_logits = detached_logits.clone()  # clone before in-place modification
            batch_indices = torch.arange(batch.shape[0])
            max_logits = torch.max(detached_logits, dim=1)[0]
            detached_logits[batch_indices, class_labels] = max_logits + 1
        decoded_batch = self.forward_decoder(detached_logits, extra_feats)
        return class_logits, extra_feats, decoded_batch

    def forward_encoder(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        latent_code = self.encoder(batch)
        class_logits, extra_feats = latent_code.split(self.num_classes, dim=1)
        return class_logits, extra_feats

    def forward_decoder(self, class_logits: Tensor, extra_feats: Tensor) -> Tensor:
        class_probabilities = F.softmax(class_logits, dim=1)
        decodable_latent_code = torch.cat((class_probabilities, extra_feats), dim=1)
        return self.decoder(decodable_latent_code)


class Discriminator(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            # nn.Sigmoid(),  # since we use BCEWithLogitsLoss
        )

    def forward(self, batch: Tensor) -> Tensor:
        return self._seq(batch)
