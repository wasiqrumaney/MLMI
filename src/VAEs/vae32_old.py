import torch
from torch import nn

Z_DIMS = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Vae32(nn.Module):
    def __init__(self, debug=False):
        super(Vae32,self).__init__()
        self.debug = debug
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1 , 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv3d(32, 64, kernel_size=2, stride=2, padding=0)
#        self.conv6 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)  
        self.fc_mu = nn.Linear(512,Z_DIMS)
        self.fc_logvar = nn.Linear(512,Z_DIMS)

        self.upconv1 = nn.ConvTranspose3d(8 ,64, kernel_size=3, stride=2, padding=0)
        self.upconv2 = nn.ConvTranspose3d(64,64, kernel_size=2, stride=1, padding=0)
        self.upconv3 = nn.ConvTranspose3d(64,32, kernel_size=3, stride=2, padding=0)
        self.upconv4 = nn.ConvTranspose3d(32,16, kernel_size=2, stride=1, padding=0)
        self.upconv5 = nn.ConvTranspose3d(16,16, kernel_size=3, stride=2, padding=0)
        self.upconv6 = nn.ConvTranspose3d(16,1, kernel_size=4, stride=1, padding=0)
#        self.conv9   = nn.Conv3d(8,1, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        y = self.relu(self.conv1(x))
        if self.debug: print('y', y.size())
        y = self.relu(self.conv2(y))
        if self.debug: print('y', y.size())
        y = self.relu(self.conv3(y))
        if self.debug: print('y', y.size())
        y = self.relu(self.conv4(y))
        if self.debug: print('y', y.size())
        y = self.relu(self.conv5(y))
        if self.debug: print('y', y.size())
        y = y.view(-1, 512)
        mu = self.fc_mu(y)
        logvar = self.fc_logvar(y)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        out = z.view(-1, 8, 2, 2, 2)
        if self.debug: print('z view',out.size())
        out = self.relu(self.upconv1(out))
        if self.debug: print('out',out.size())
        out = self.relu(self.upconv2(out))
        if self.debug: print('out',out.size())
        out = self.relu(self.upconv3(out))
        if self.debug: print('out',out.size())
        out = self.relu(self.upconv4(out))
        if self.debug: print('out',out.size())
        out = self.relu(self.upconv5(out))
        if self.debug: print('out',out.size())
#        out = self.relu(self.upconv6(out))
        out = self.sigmoid(self.upconv6(out))
        return out
    
    def forward(self,x):
        if self.debug: print('input',x.size())
        mu, logvar = self.encode(x)
        if self.debug: print('logvar',logvar.size())
        z = self.reparameterize(mu, logvar)
        if self.debug:print('z',z.size())
        out = self.decode(z)
        if self.debug: print('out',out.size())
        return out, mu, logvar
