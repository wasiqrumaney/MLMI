import torch


class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.cube_len = args.cube_len

        padd = (0, 0, 0)
#         if self.cube_len == 32:
#                 padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.args.z_size, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.ReLU(inplace = True)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.ReLU(inplace = True)
            #torch.nn.LeakyReLU(self.args.leak_value)
        )
                
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*1, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            #torch.nn.BatchNorm3d(1), #lets try opening this later 
            #torch.nn.Linear(-1,1),
            torch.nn.ReLU(),
            #torch.nn.LeakyReLU(self.args.leak_value),
            torch.nn.Tanh() ###
            #torch.nn.Sigmoid()
        )
      

    def forward(self, x):
        out = x.view(-1, self.args.z_size, 1, 1, 1)
        #print("start",out.size()) # torch.Size([64, 100, 1, 1, 1])
        out = self.layer1(out)     
        #print("G:L1",out.size())  #torch.Size([64, 512, 4, 4, 4])
        out = self.layer2(out)
        #print("G:L2",out.size()) # torch.Size([64, 256, 8, 8, 8])
        out = self.layer3(out)
        #print("G:L3",out.size()) # torch.Size([64, 128, 16, 16, 16])
        out = self.layer4(out)
        #print("G:L4",out.size()) #  torch.Size([64, 64, 32, 32, 32])
        out = self.layer5(out)
        #print("G:L5",out.size()) #  torch.Size([64, 1, 64, 64, 64])
        out = out.view(-1, self.cube_len*self.cube_len*self.cube_len)
        #print("G:Final",out.size()) # torch.Size([64, 32768])

        return out


class _D(torch.nn.Module):
    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args
        self.cube_len = args.cube_len

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.LayerNorm([64, 32, 32, 32]),
            torch.nn.LeakyReLU(self.args.leak_value)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
	    torch.nn.LayerNorm([128, 16, 16, 16]),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.LayerNorm([256, 8, 8, 8]),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.LayerNorm([512, 4, 4, 4]),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=args.bias, padding=(0,0,0)),
            #torch.nn.LeakyReLU(self.args.leak_value)
            #torch.nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)
        #print("start",out.size()) #  torch.Size([64, 1, 64, 64, 64])
        out = self.layer1(out)
        #print("D:L1",out.size()) #  torch.Size([64, 64, 32, 32, 32])
        out = self.layer2(out)
        #print("D:L2",out.size()) #  torch.Size([64, 128, 16, 16, 16])
        out = self.layer3(out)
        #print("D:L3",out.size()) #  torch.Size([64, 256, 8, 8, 8])
        out = self.layer4(out)
        #print("D:L4",out.size()) #  torch.Size([64, 512, 4, 4, 4])
        out = self.layer5(out)
        #print("D:L5",out.size()) #  torch.Size([64, 1, 1, 1, 1])
        out = out.view(-1,1) 
        #print("final",out.size()) # torch.Size([64, 1])
        return out
