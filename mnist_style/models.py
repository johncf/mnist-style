import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self._seq(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 32, 7, 7)
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv2(x))
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self._seq(x)
