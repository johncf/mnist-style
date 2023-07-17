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
