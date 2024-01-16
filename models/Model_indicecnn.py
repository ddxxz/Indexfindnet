import torch
import torch.nn as nn
import numpy as np
import sys

from models.BaseModel import BaseModel
import torch.nn.functional as F
#from scipy import signal
from einops import rearrange, repeat
#from models.GRCNN import *
#from models.Model_grconv import Gatetransformer_Block


class GateConv1D(nn.Module):
    def __init__(self,in_channels,out_channels,dilation,pad=0,s=1):
        super(GateConv1D, self).__init__()
        #pad = 0 #5//2 * dilation
        self.conv = nn.Conv1d(in_channels,out_channels*2,kernel_size=5,dilation=dilation,stride=s,padding=pad)
    def forward(self,x):
        x = self.conv(x)
        main,gate = torch.chunk(x,2,1)
        x = main *gate.sigmoid()
        return x

def ref_shuffle( x):
    batchsize, num_channels, num_ref = x.data.size()
    #print(x.shape)
    assert (num_ref % 2 == 0)
    x = x.reshape(batchsize , num_channels , num_ref//2,2)
    x = x.permute(0, 1, 3,2)
    x = x.reshape(batchsize,  num_channels , num_ref)
    return x


class Indice_add_mul(nn.Module):#0.874
    def __init__(self, indim,outdim,dilation):
        super(Indice_add_mul, self).__init__()
        hidden = outdim #*4#  1,2,3,4,5
        k = 5
        pad = k // 2 *dilation
        # k = 10 if dilation == 2 else 5
        # pad = k // 2 if k == 10 else 0
        #print(k,pad)
        self.Wq = nn.Sequential(  #
            nn.Conv1d(indim, hidden,bias=False,kernel_size=k,dilation=dilation,padding=pad),#1,0,0,0,0 -> 1
            nn.BatchNorm1d(num_features=hidden),
            nn.Sigmoid()
        )
        self.Wk = nn.Sequential(
            nn.Conv1d(indim, hidden,bias=False,kernel_size=k,dilation=dilation,padding=pad),#0,0,0,0,1 -> 5
            nn.BatchNorm1d(num_features=hidden),
            nn.Sigmoid()
        )
        self.Wv = nn.Sequential(
            nn.Conv1d(indim, hidden,bias=False,kernel_size=k,dilation=dilation,padding=pad),#10
            nn.BatchNorm1d(num_features=hidden),
            nn.Sigmoid()
        )
        self.Wa = nn.Sequential(
            nn.Conv1d(indim, hidden,bias=False,kernel_size=k,dilation=dilation,padding=pad),
            nn.BatchNorm1d(num_features=hidden),
            nn.Sigmoid()
        )
        self.output = nn.Sequential(nn.Conv1d(hidden,outdim,kernel_size=1),nn.PReLU())

    def forward(self, inputs):
        #print(inputs.shape)
        #print(inputs.shape)
        Q = self.Wq(inputs) #（4，128）*（128，128）=（4，128)#        16
        K = self.Wk(inputs) #（4，100,128）*（128，128）=（4，100,128) 16
        V = self.Wv(inputs)
        A = self.Wa(inputs)
        indice = (Q+K)/(torch.abs(V*A)+0.06)
        # out = torch.cat([Q,K,indice],dim=1)
        out = self.output(indice)
        return out


class IndiceCNN(BaseModel):
    def __init__(self, device,input_size):
        super(IndiceCNN, self).__init__(device)
        self.input_size = input_size
        self.avgpool = nn.AvgPool1d(kernel_size=8, stride=2, padding=2)#kernel_size=2
        self.bn0 = nn.BatchNorm1d(num_features=1)
        self.convblock1 = nn.Sequential(
            GateConv1D(1,50,1),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )
        self.convblock2 = nn.Sequential(
            Indice_add_mul(50, 50, 1),
            Indice_add_mul(50, 50, 2),
            Indice_add_mul(50, 50, 2),
            # Indice_add_mul(20, 20, 1),
            # Indice_add_mul(20, 20, 2),
            # Indice_add_mul(20, 20, 2),

            nn.AvgPool1d(2),
        )
        if input_size==204:
            self.fc1 = nn.Linear(in_features=1200,out_features=100)#900,
        else:
            self.fc1 = nn.Linear(in_features=700, out_features=1000)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=100, out_features=1)

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.bn0(x)
        x = self.convblock1(x)
        x = self.convblock2(x)# b,c,t
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#---------------------------用高光谱图像空间特征提取rectangle 4,1  CNN+attention output 1,1------------------------------


if __name__ == "__main__":
    #from torchsummary import summary
    # x = torch.randn([1,1,204])
    # net = indice_grconv(None, 204,2,1)
    # out = net(x)
    x1 = torch.randn([8, 1, 204])
    x2 = torch.randn([1, 204, 512, 1])
    # net = CNN_grconv_mul(None, 204)
    net = IndiceCNN(None, 204)
    #net =Himg_add_mul_gatecnn(None,4)
    #net = Himg_cnnformer_gate_indice(None,204,2,1)
    #net = Himg_cnnformer_indice(None, 4,2,1)
    out = net(x1)
    #summary(net,x2.shape[1:],device='cpu')
    from ptflops import get_model_complexity_info
    import re
    macs,params = get_model_complexity_info(net,tuple(x1.shape[1:]),as_strings=True,
                                            print_per_layer_stat=False,verbose=True)

    flops = eval(re.findall(r'([\d.]+)',macs)[0])*2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)',macs)[0][0]

    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))
