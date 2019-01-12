import torch


class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.cube_len = args.cube_len

        padd = (0, 0, 0)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.args.z_size, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.ReLU(inplace = True)
            #torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.ReLU(inplace = True)
            #torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(1),
            #torch.nn.Linear(-1,1),
            torch.nn.ReLU(),
            #torch.nn.LeakyReLU(self.args.leak_value),
            torch.nn.Tanh() ###
            #torch.nn.Sigmoid()
        )
      

    def forward(self, x):
        out = x.view(-1, self.args.z_size, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, self.cube_len*self.cube_len*self.cube_len)

        return out


class _D(torch.nn.Module):
    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args
        self.cube_len = args.cube_len

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)


        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            #torch.nn.LayerNorm([64, 8, 8, 8]),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            #torch.nn.LayerNorm([128, 4, 4, 4]),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=args.bias, padding=padd),
            #torch.nn.LeakyReLU(self.args.leak_value)
            #torch.nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(-1,1) 
        return out