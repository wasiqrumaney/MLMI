import torch
from torch import nn
ENC_OUTSHAPE = 512
DEC_INSHAPE = (1, 8, 8, 8)

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
    def __init__(self,  z_dims, nf=8, debug=False, device="cpu"):
        super(Vae32, self).__init__()
        self.debug = debug
        self.device = device

        self.z_dims = z_dims
        self.encoder = nn.Sequential(                                   # 32
            nn.Conv3d(1, nf, kernel_size=3, stride=2, padding=0),       # 15
            nn.ReLU(True),
            nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=0),      # 13
            nn.ReLU(True),
            nn.Conv3d(nf, nf*2, kernel_size=3, stride=2, padding=0),    # 6
            nn.ReLU(True),
            nn.Conv3d(nf*2, nf*2, kernel_size=3, stride=1, padding=0),  # 4
            nn.ReLU(True),
            nn.Conv3d(nf*2, nf*2, kernel_size=3, stride=1, padding=0),     # 2
            Flatten()                                                   # 2*2*2 = 8 * nf*2
        )
        self.fc_mu = nn.Linear(ENC_OUTSHAPE, self.z_dims)
        self.fc_logvar = nn.Linear(ENC_OUTSHAPE, self.z_dims)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dims, ENC_OUTSHAPE),
            Reshape(DEC_INSHAPE),
            nn.ConvTranspose3d(1, nf*8, kernel_size=4, stride=2, padding=0),    # 6
            nn.ReLU(True),
            nn.ConvTranspose3d(nf*8, nf*4, kernel_size=3, stride=1, padding=0), # 8
            nn.ReLU(True),
            nn.ConvTranspose3d(nf*4, nf*4, kernel_size=4, stride=2, padding=0), # 18
            nn.ReLU(True),
            nn.ConvTranspose3d(nf*4, nf*2, kernel_size=3, stride=1, padding=0), # 20
            nn.ReLU(True),
            nn.ConvTranspose3d(nf*2, nf*2, kernel_size=4, stride=1, padding=0), # 23
            nn.ReLU(True),
            nn.ConvTranspose3d(nf*2, nf, kernel_size=4, stride=1, padding=0),   # 26
            nn.ReLU(True),
            nn.ConvTranspose3d(nf, nf, kernel_size=4, stride=1, padding=0),     # 29
            nn.ReLU(True),
            nn.ConvTranspose3d(nf, 1, kernel_size=4, stride=1, padding=0),      # 32
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
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar