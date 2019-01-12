import torch
from torch import nn

Z_DIMS = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# VAE Model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()  # input 1x16x16x16
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=2, stride=1, padding=1)  # 16x16x16x16
        self.conv2 = nn.Conv3d(1, 64, kernel_size=2, stride=2, padding=0)  # 64x8x8x8
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)  # 64x8x8x8
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)  # 64x8x8x8
        self.fc1 = nn.Linear(64 * 8 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2 * Z_DIMS)

        self.fc3 = nn.Linear(Z_DIMS, Z_DIMS)
        self.fc4 = nn.Linear(Z_DIMS, 64 * 8 * 8 * 8)
        self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(-1, 32768)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        mu = x[:, :Z_DIMS]
        logvar = x[:, Z_DIMS:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        out = self.relu(self.fc3(z))
        out = self.relu(self.fc4(out))
        # reshape
        out = out.view(-1, 64, 8, 8, 8)
        out = self.relu(self.upconv1(out))
        out = self.relu(self.upconv2(out))
        out = self.relu(self.upconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar