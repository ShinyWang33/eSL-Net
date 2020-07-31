import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from math import sqrt
import torch.nn.functional as F
import numpy as np
import math
import scipy.io as sio

########################################################################
##########################################################################
class MySCN(nn.Module):
    def __init__(self):
        super(MySCN, self).__init__()
        self.W1 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.S1 = nn.Conv2d(128, 64, 3, 1, 1, groups=1, bias=False)
        self.S2 = nn.Conv2d(64, 128, 3, 1, 1, groups=1, bias=False)
        self.shlu = nn.ReLU(True)

    def forward(self, input):
        x1 = input[:,range(0,64),:,:]
        event_input = input[:,range(64,128),:,:]

        x1 = torch.mul(x1,event_input)
        z = self.W1(x1)
        tmp = z
        for i in range(25):
            ttmp = self.shlu(tmp)
            x = self.S1(ttmp)
            x = torch.mul(x,event_input)
            x = torch.mul(x,event_input)
            x = self.S2(x)
            x=ttmp-x
            tmp = torch.add(x, z)
        c = self.shlu(tmp)
        return c

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.scn = nn.Sequential(MySCN())
        self.image_d=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.shu1=nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.shu2=nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.endconv=nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.ps1=nn.PixelShuffle(2)
        self.ps2=nn.PixelShuffle(2)

        '''for p in self.parameters():
                    p.requires_grad=False'''

        self.event_c1=nn.Conv2d(in_channels=40, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.event_c2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))


    def forward(self, x):
        x1 = x[:,range(0,1),:,:]
        x1=self.image_d(x1)

        event = x[:,range(1,41),:,:]
        event_out=self.event_c1(event)
        event_out=torch.sigmoid(event_out)
        event_out=self.event_c2(event_out)
        event_out=torch.sigmoid(event_out)

        scn_input=torch.cat([x1,event_out],1)
        
        out = self.scn(scn_input)
		
        out=self.shu1(out)
        out=self.ps1(out)
        out=self.shu2(out)
        out=self.ps2(out)
		
        out=self.endconv(out)
		
        return out


if __name__ == '__main__':
    print('running model.py')
    from tensorboardX import SummaryWriter
    import argparse
    parser = argparse.ArgumentParser(description="rlcsc_graph")
    model = Net(parser.parse_args())

    with SummaryWriter(comment='net') as w:
        w.add_graph(model,(torch.Tensor(48,1,33,33),))
