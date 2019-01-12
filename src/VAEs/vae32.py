import torch
from torch import nn
ENC_OUTSHAPE = 512
DEC_INSHAPE = (8, 2, 2, 2)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

class Vae32(nn.Module):
    def __init__(self, z_dims, debug=False, device="cpu"):
        super(Vae32, self).__init__()
        self.debug = debug
        self.device = device

        self.z_dims = z_dims
        self.encoder = nn.Sequential(                                   # 32
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=0),       # 15
            nn.ReLU(True),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0),      # 13
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0),    # 6
            nn.ReLU(True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0),  # 4
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=2, stride=2, padding=0),     # 2
            Flatten()                                                   # 2*2*2 = 8
        )
        self.fc_mu = nn.Linear(ENC_OUTSHAPE, self.z_dims)
        self.fc_logvar = nn.Linear(ENC_OUTSHAPE, self.z_dims)
        self.decoder = nn.Sequential(
            Reshape(DEC_INSHAPE),
            nn.ConvTranspose3d(8, 64, kernel_size=3, stride=2, padding=0),    # 6
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=1, padding=0), # 8
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=0), # 18
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=1, padding=0), # 20
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 16, kernel_size=3, stride=2, padding=0), # 23
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=1, padding=0),   # 26
            nn.Sigmoid()
        )

    def encode(self, x):
        y = x
        if self.debug: print(y.size())
        for module in self.encoder:
            y = module(y)
            if self.debug: print(y.size())
        mu = self.fc_mu(y)
        logvar = self.fc_logvar(y)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.device)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        rec = z
        if self.debug: print(rec.size())
        for module in self.decoder:
            rec = module(rec)
            if self.debug: print(rec.size())
        return rec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar