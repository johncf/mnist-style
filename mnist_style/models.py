import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_groups=2, **conv2d_args):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, bias=False, **conv2d_args),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(True),
        )


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._seq = nn.Sequential(
            ConvBlock(1, 16, kernel_size=5, stride=2, padding=2, norm_groups=4),
            ConvBlock(16, 32, kernel_size=3, stride=1, padding=1, norm_groups=8),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self._seq(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.conv1 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1, norm_groups=8)
        self.conv2 = ConvBlock(32, 4, kernel_size=3, stride=1, padding=1, norm_groups=2)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 32, 7, 7)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            # nn.Sigmoid(),  # since we use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self._seq(x)
