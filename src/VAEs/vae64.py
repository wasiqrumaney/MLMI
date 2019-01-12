#import numpy as np
import torch
from torch import nn

Z_DIMS = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#k = 64

class Vae64(nn.Module):
    def __init__(self):
        super(Vae64,self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1 , 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=0)
        self.conv6 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv7 = nn.Conv3d(64, 1 , kernel_size=1, stride=1, padding=0)       
        self.fc_mu = nn.Linear(27,Z_DIMS)
        self.fc_logvar = nn.Linear(27,Z_DIMS)
        
        self.upconv1 = nn.ConvTranspose3d(1 ,64, kernel_size=4, stride=2, padding=0)
        self.upconv2 = nn.ConvTranspose3d(64,64, kernel_size=4, stride=1, padding=0)
        self.upconv3 = nn.ConvTranspose3d(64,32, kernel_size=4, stride=2, padding=0)
        self.upconv4 = nn.ConvTranspose3d(32,16, kernel_size=3, stride=1, padding=0)
        self.upconv5 = nn.ConvTranspose3d(16,16, kernel_size=4, stride=2, padding=0)
        self.upconv6 = nn.ConvTranspose3d(16,1, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        
    def encode(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.relu(self.conv4(y))
        y = self.relu(self.conv5(y))
        y = self.relu(self.conv6(y))
        y = self.relu(self.conv7(y))
#        print('y', y.size())
        y = y.view(-1, 27)
        mu = self.fc_mu(y)
        logvar = self.fc_logvar(y)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        eps = torch.randn([mu.size()[0], Z_DIMS]).to(device)
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        z = z.view(-1, 1, 4, 4, 4)
#        print('z view',z.size())
        out = self.relu(self.upconv1(z))
        out = self.relu(self.upconv2(out))
        out = self.relu(self.upconv3(out))
        out = self.relu(self.upconv4(out))
        out = self.relu(self.upconv5(out))
        out = self.sigmoid(self.upconv6(out))
        return out
    
    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar
