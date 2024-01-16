# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/10 9:59
@Auth ： dengxz
@File ：Model_indexfind.py
@IDE ：PyCharm
"""
import torch.nn as nn
import numpy as np
import torch
import sys
from pathlib import Path
from models.BaseModel import BaseModel
import torch.nn.functional as F
sys.path.append("/mnt/e/deep_learning/Indexfindnet")
from models.unet.unet_model import *
from models.unet.unet_parts import *
from models.gumbelmodule import GumbelSoftmaxV3 #Gumbel_Softmax,GumbleSoftmax,GumbelSoftmaxV2,
class AttentionV2(nn.Module):
    def __init__(self,dilation,C =50,len=204,onehot=False):
        super(AttentionV2, self).__init__()
        k = 5
        pad = k // 2 * dilation
        self.Wq = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=C,bias=False,kernel_size=k,dilation=dilation,padding=pad),
                                nn.BatchNorm1d(num_features=C),
                                nn.ELU()
                                #nn.ReLU(),

                                )

        self.Wv = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C,bias=False,kernel_size=k,dilation=dilation*2,padding=pad*2),
                                # nn.BatchNorm1d(num_features=C),
                                # nn.ReLU()
                                )
       # self.linear = nn.Linear(len,len)

        self.sigmoid = nn.Sigmoid()
        self.onehot = GumbelSoftmaxV3(len)##GumbleSoftmax()
        self.use_onehot = onehot
        self.C = C
        #self.bs = bs
    def forward(self, inputs, ref=None,return_weight=False):
        #print(inputs.shape)#B , T , C
        Q = self.Wq(inputs) #8,50,101
        O = self.Wv(Q) #8,50,101
        if self.use_onehot:
            O = self.onehot(O,1,True)

        else:
            O=torch.softmax(O,dim=1)#8,101,50self.sigmoid(O) #8,64,204
            print('O',O[0,0])
            #0.9,

        A_ = O.transpose(1, 2)#8,101,50
    
        #O = torch.matmul(K.to('cuda'),A_.to('cuda'))/np.sqrt(self.C)
        O = torch.matmul(inputs.to('cuda'),A_.to('cuda'))/np.sqrt(self.C)#8,50
        
        O = torch.squeeze(O, dim=-1)
        #O= self.layernorm(O)
        O = self.sigmoid(O)
        #O = V * A_
        if return_weight:
            return O,A_
        else:
            return O

#BN  kernel-size dilation  Gatedconv
class Attention(nn.Module):
    def __init__(self,dilation,C =50):
        super(Attention, self).__init__()

        k = 5
        pad = k // 2 * dilation
        self.Wq = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=1,bias=False,kernel_size=k,dilation=dilation,padding=pad),
                                nn.BatchNorm1d(num_features=1),
                                nn.ReLU(),
                                # nn.Conv1d(in_channels=10, out_channels=1, bias=False, kernel_size=k, dilation=dilation,
                                #           padding=pad),
                                # nn.BatchNorm1d(num_features=1),
                                # nn.ReLU(),
                                )
        # self.Wk = nn.Conv1d(in_channels=50, out_channels=50,bias=False,kernel_size=k,dilation=dilation,padding=pad)
        # self.qk = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=1,bias=False,kernel_size=k,dilation=dilation,padding=pad),
        #                         nn.BatchNorm1d(num_features=1),
        #                         )

        self.Wv = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C,bias=False,kernel_size=k,dilation=dilation,padding=pad),
                                nn.BatchNorm1d(num_features=C),
                                nn.ReLU(),
                                # nn.Conv1d(in_channels=50, out_channels=50, bias=False, kernel_size=k, dilation=dilation,
                                #           padding=pad),
                                # nn.BatchNorm1d(num_features=50),
                                # nn.ReLU(),
                                )
        #self.layernorm = nn.LayerNorm(50)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs, return_weight=False):
        #print(inputs.shape)#B , T , C
        Q = self.Wq(inputs) #8,1,101
        #print('Q',Q.shape)
        # K = self.Wk(inputs) #8,50,101
        V = self.Wv(inputs) #8,50,101
        #print('V', V.shape)
        #A=torch.matmul(K.transpose(1,2),Q) #
        Q = Q.transpose(1, 2)
        A_=torch.softmax(Q,dim=1)#8,101,1

        #V = V.transpose(1, 2)#8,101,50
        O=torch.matmul(V,A_)/np.sqrt(50) #_

        O = torch.squeeze(O, dim=-1)
        #O= self.layernorm(O)
        O = self.sigmoid(O)
        #O = V * A_
        if return_weight:
            return O,A_
        else:
            return O
#---------------------------仅用高光谱数据1d  bnCNN------------------------------
class Add_mul(nn.Module):#0.874
    def __init__(self):
        super(Add_mul, self).__init__()
        #self.output = nn.Sequential(nn.Linear(50, 50), nn.PReLU())
    def forward(self,Q,K,V,A):
        indice = (Q+K)/(torch.abs(V*A)+0.06)

        #indice = self.output(indice)
        return indice

class Index_Add_mul(nn.Module):
    def __init__(self,C,out,len=204,v2=False,onehot=False):
        super(Index_Add_mul, self).__init__()
        if v2:
            self.ref1=AttentionV2(2,C,len=len,onehot=onehot)
            self.ref2 = AttentionV2(2,C,len=len,onehot=onehot)
            self.ref3 = AttentionV2(2,C,len=len,onehot=onehot)
            self.ref4 = AttentionV2(2,C,len=len,onehot=onehot)

        else:
            self.ref1=Attention(2,C,len=len)
            self.ref2 = Attention(2,C,len=len)
            self.ref3 = Attention(2,C,len=len)
            self.ref4 = Attention(2,C,len=len)

        self.index = Add_mul()
        self.linear = nn.Sequential(nn.Linear(C, out), nn.PReLU())
        self.len =len

    def forward(self, inputs,return_weight=False):
        if return_weight:
            B1, W1 = self.ref1(inputs,return_weight=return_weight)
            B2, W2 = self.ref2(inputs,return_weight=return_weight)
            B3, W3 = self.ref3(inputs, return_weight=return_weight)
            B4, W4 = self.ref4(inputs, return_weight=return_weight)
        else:
            B1 = self.ref1(inputs)
            B2 = self.ref2(inputs)
            B3 = self.ref3(inputs)
            B4 = self.ref4(inputs)
        # print('B1',B1.shape)
        #print(B1.is_cuda,B2.is_cuda,B3.is_cuda,B4.is_cuda)
        x = self.index(B1, B2, B3, B4)
        #print(inputs.is_cuda)
        x = self.linear(x)
        if return_weight:
            return x, [W1, W2, W3, W4]
        else:
            return x
#---------------------------仅用高光谱数据  CNN------------------------------


#---------------------------仅用高光谱数据1d  bnCNN------------------------------
class Sub_sub(nn.Module):#0.874
    def __init__(self):
        super(Sub_sub, self).__init__()
    def forward(self,Q,K,V,A):
        indice = (Q-K)/(torch.abs(V-A)+0.06)
        return indice

class Index_Sub_sub(nn.Module):
    def __init__(self,C,out,len=204,v2=False,onehot=False):
        super(Index_Sub_sub, self).__init__()
        if v2:
            self.ref1 = AttentionV2(2,C,len=len,onehot=onehot)
            self.ref2 = AttentionV2(2,C,len=len,onehot=onehot)
            self.ref3 = AttentionV2(2,C,len=len,onehot=onehot)
            self.ref4 = AttentionV2(2,C,len=len,onehot=onehot)

        else:
            self.ref1 = Attention(2,C,len=len)
            self.ref2 = Attention(2,C,len=len)
            self.ref3 = Attention(2,C,len=len)
            self.ref4 = Attention(2,C,len=len)
        self.index = Sub_sub()
        self.linear = nn.Sequential(nn.Linear(C, out), nn.PReLU())
        self.len =len

    def forward(self, inputs, return_weight=False):
        # print('inputs',inputs.shape)
        if return_weight:
            B1, W1 = self.ref1(inputs,return_weight=return_weight)
            B2, W2 = self.ref2(inputs,return_weight=return_weight)
            B3, W3 = self.ref3(inputs, return_weight=return_weight)
            B4, W4 = self.ref4(inputs, return_weight=return_weight)
        else:
            B1 = self.ref1(inputs)
            B2 = self.ref2(inputs)
            B3 = self.ref3(inputs)
            B4 = self.ref4(inputs)
        # print('B1',B1.shape)
        x = self.index(B1, B2, B3, B4)
        # print('x',x.shape)
        x = self.linear(x)
        if return_weight:
            return x, [W1, W2, W3, W4]
        else:
            return x
#---------------------------仅用高光谱数据  CNN------------------------------

#---------------------------仅用高光谱数据1d  bnCNN------------------------------
class Div(nn.Module):#0.874
    def __init__(self):
        super(Div, self).__init__()
    def forward(self,Q,K):
        indice = Q/(K+0.06)
        return indice

class Index_Div(nn.Module):
    def __init__(self,C,out,len=204,v2=False,onehot=False):
        super(Index_Div, self).__init__()
        if v2:
            self.ref1=AttentionV2(2,C,len=len,onehot=onehot)
            self.ref2 = AttentionV2(2,C,len=len,onehot=onehot)
        else:
            self.ref1=Attention(2,C,len=len)
            self.ref2 = Attention(2,C,len=len)
        self.index = Div()
        self.linear = nn.Sequential(nn.Linear(C, out), nn.PReLU())
        self.len =len

    def forward(self, inputs,return_weight=False):
        if return_weight:
            B1, W1 = self.ref1(inputs, return_weight=return_weight)
            B2, W2 = self.ref2(inputs, return_weight=return_weight)
        else:
            B1 = self.ref1(inputs)
            B2 = self.ref2(inputs)
        # print('B1',B1.shape)
        x = self.index(B1, B2)
        # print('x',x.shape)
        x = self.linear(x)
        if return_weight:
            return x, [W1, W2]
        else:
            return x
#---------------------------仅用高光谱数据  CNN------------------------------

#---------------------------仅用高光谱数据  CNN------------------------------
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


class IndexfindNet(BaseModel):
    def __init__(self, device,inputlen):
        super(IndexfindNet, self).__init__(device)
        #self.avgpool = nn.AvgPool1d(kernel_size=8, stride=4, padding=2)  # kernel_size=2
        self.device = device
        self.channel = 64
        self.bn0 = nn.BatchNorm1d(num_features=1)
        self.conv1=nn.Sequential(
                GateConv1D(1,self.channel,1,pad=2),
                #nn.Conv1d(1,self.channel,5,padding=2),
                nn.BatchNorm1d(num_features=self.channel),
                nn.ReLU(),
                #nn.AvgPool1d(2),
                                 )
        self.unet = UNet(n_channels=self.channel, n_classes=1,encs=[self.channel,self.channel,self.channel,self.channel], 
                         bilinear=True)
        #self.Indexfind1 = Index_Sub_mul(64,64,v2=True,onehot=True)
        self.Indexfind2 = Index_Add_mul(self.channel,self.channel,len=inputlen,v2=True,onehot=True)
        self.Indexfind3 = Index_Sub_sub(self.channel,self.channel,len=inputlen,v2=True,onehot=True)
        #self.Indexfind4 = Index_Sub_add(64,64,v2=True,onehot=True)
        self.Indexfind5 = Index_Div(self.channel,self.channel,len=inputlen,v2=True,onehot=True)
        #from models.transformer_torch import TransformerEncoderLayer
        #self.transformer = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=64 * 4)#batch_first= False  T,B,C
        #self.transformer = nn.TransformerEncoderLayer(d_model=50, nhead=5, dim_feedforward=50 * 4,batch_first=True)
        #self.channelattention = ChannelDeepTimeSenseSELayer(3)
        self.att = nn.Sequential(nn.Conv1d(self.channel,self.channel,1,bias=False),
                                 #nn.Conv1d(self.channel,self.channel,1,bias=False),
                                # nn.BatchNorm1d(64),
                                 #nn.ReLU()
                                 )
        self.fc1 = nn.Linear(self.channel,128)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        #self.softmax = GumbelSoftmaxV3(3)
    
    def att_loss(self,all_weights):
        loss = 0
        for func_weights in all_weights:
            for idx_weights in func_weights:
                idx = torch.randperm(len(idx_weights))
                #print(idx_weights.shape)8,204,64
                b,t,c = idx_weights.shape
                # idx_weights = idx_weights.transpose(-1,-2)
                # rand_weights = idx_weights[idx]
                # loss_w = F.cross_entropy(idx_weights.reshape(b*c,t),rand_weights.reshape(b*c,t))
                
                v = idx_weights * idx_weights[idx]                
                #v = torch.prod(idx_weights,dim=1)
                loss_b = torch.sum(v,dim=1) 
                loss_b = - torch.mean(loss_b,dim=-1)

                idx = torch.randperm(c)
                loss_c = torch.sum(idx_weights * idx_weights[...,idx],dim=1)
                loss_c = torch.mean(loss_c,dim=-1)
                loss += loss_b + loss_c
        #print(loss)
        return  torch.mean(loss,dim=0) / 10.0
    def forward(self, inputs,return_weight=False):
        #print('inputs',inputs.shape)
        #x = self.avgpool(inputs)
        device = torch.device('cuda:0')
        inputs = inputs.to(device)
        #print('inputs',torch.max(inputs),torch.min(inputs))
        #print(inputs.is_cuda)
        x = self.bn0(inputs)
        #print('bn0',torch.max(x),torch.min(x))
        
        x = self.conv1(x)
        #print('x',x[0].max(dim=-1))

        m = self.unet(x)
        masked = inputs * m
        #print('masked',torch.max(masked),torch.min(masked))
        #masked = masked
        # print('in',inputs[0,0])
        # print('m',m[0,0])
        #print(x.shape)
        #x = x.transpose(1, 2)
        #print('conv', x.shape)
        #x1,A1 = self.Indexfind1(x,return_weight=True)
        x2,A2 = self.Indexfind2(masked,return_weight=True)
        x3,A3 = self.Indexfind3(masked,return_weight=True)
        # x4,A4 = self.Indexfind4(x,return_weight)
        x5,A5 = self.Indexfind5(masked,return_weight=True)
        #x6, A6 = self.Indexfind6(x, return_weight)

        #print(x.shape)
        x = torch.cat([x2,x3,x5],dim=1)
        #print(x.shape)#b,c,t
        #x,attention_weight = self.transformer(x)#不设置batchfirst输入只能是tbc
        #x,attention_weight = self.channelattention(x)
        #x = torch.mean(x,dim=1,keepdim=True)
        att = self.att(x.transpose(1,2))#b,c,3
        #att = torch.mean(att,dim=-1,keepdim=True)

        att = att.transpose(1,2) # b,3,c
        att = torch.softmax(att,dim=1)

        #att = self.softmax(att,1,True).transpose(1,2)
                          
        #print(att[0,:,0])
        #print(x.shape,att.shape)
        x = torch.sum(x *att,dim=1,keepdim=True)
        
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #print(x.shape)
        if return_weight:
            return x,[A2,A3,A5],att,masked
        else:
            if self.training:
                #loss = self.att_loss([A2,A3,A5])
                sdr_loss = self.sdr_loss(inputs,masked)
                return x, sdr_loss
            return x
    def sdr_loss(self,inp,label):
        w = torch.sum(inp * label,dim=-1)/ (torch.norm(inp,p=2,dim=-1) * torch.norm(label,p=2,dim=-1) + 1e-12)
        w = torch.mean(w)
        return - w


#---------------------------仅用高光谱数据  CNN------------------------------

if __name__ == "__main__":
    #from torchsummary import summary

    x1 = torch.randn([7,1,204])
    x2 = torch.randn([1, 204, 512, 1])
    x3 = torch.randn([7,4,204])
    low = 0.0
    high = 1.0

# 范围缩放的线性变换
    scaled_values = 0.5 * (torch.randn(7,1,204) + 1.0)
    scaled_values.clamp_(low, high)  # 将结果限制在 [0, 1) 范围内

    print(torch.max(scaled_values),torch.min(scaled_values))
    #net = IndexfindNet_transformer_himg(None,4)
    #net = IndexfindNet_transformer_weight_gl(None,4,204)
    net = IndexfindNet(None,204)
    device = torch.device('cuda:0')
    net.to(device)
    # USE_CUDA = torch.cuda.is_available()
    # x1=x1.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
    #print(torch.cuda.current_device())
    out = net(scaled_values)


#多种指数计算并行，接transformer。
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

