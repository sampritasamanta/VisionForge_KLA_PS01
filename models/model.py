import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c,c,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):
        return x + self.net(x)

class ResidualUNet(nn.Module):
    def __init__(self, base_channels=32, num_res_blocks=4):
        super().__init__()
        c=base_channels
        self.i=nn.Conv2d(1,c,3,padding=1)
        self.e1=nn.Sequential(ResidualBlock(c),ResidualBlock(c))
        self.d1=nn.Conv2d(c,2*c,3,stride=2,padding=1)
        self.e2=nn.Sequential(ResidualBlock(2*c),ResidualBlock(2*c))
        self.d2=nn.Conv2d(2*c,4*c,3,stride=2,padding=1)
        self.b=nn.Sequential(*[ResidualBlock(4*c) for _ in range(num_res_blocks)])
        self.u2=nn.Conv2d(4*c,2*c,3,padding=1)
        self.x2=nn.Sequential(ResidualBlock(2*c),ResidualBlock(2*c))
        self.u1=nn.Conv2d(2*c,c,3,padding=1)
        self.x1=nn.Sequential(ResidualBlock(c),ResidualBlock(c))
        self.o=nn.Conv2d(c,1,3,padding=1)
    def forward(self,x,target_size=None):
        a=self.i(x); e1=self.e1(a)
        e2=self.e2(self.d1(e1)); b=self.b(self.d2(e2))
        z=F.interpolate(b,size=e2.shape[-2:],mode="bilinear",align_corners=False)
        z=self.x2(self.u2(z)+e2)
        z=F.interpolate(z,size=e1.shape[-2:],mode="bilinear",align_corners=False)
        z=self.x1(self.u1(z)+e1)
        y=self.o(z)
        return y if target_size is None else F.interpolate(y,size=target_size,mode="bilinear",align_corners=False)
