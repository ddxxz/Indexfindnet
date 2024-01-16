import torch
import torch.nn as nn
import numpy as np
import sys

from models.BaseModel import BaseModel
#from BaseModel import BaseModel
import torch.nn.functional as F
#from scipy import signal


class SG(nn.Module):
    def __init__(self):
        super(SG, self).__init__()
        coef = signal.savgol_coeffs(21, 2,deriv=0, delta=1.0)
        coefs = torch.tensor(coef,dtype=torch.float32)
        coefs = coefs[None,None,]
        #print(coefs.shape)
        self.params=nn.Parameter(coefs,requires_grad=False)
        #2d  c_in,c_out, w,h  4, 8,3,3
        #id  c_in,c_out, w    4,8,3           1,1,21
    def forward(self, x):

        x = F.conv1d(x,self.params,padding='same')
        return x


#---------------------------仅用高光谱数据1d  bnCNN------------------------------
class OnedCNN(BaseModel):
    def __init__(self, device,input_size):
        super(OnedCNN, self).__init__(device)
        self.input_size = input_size
        self.avgpool = nn.AvgPool1d(kernel_size=10, stride=10, padding=2)#kernel_size=2
        self.bn0 = nn.BatchNorm1d(num_features=1)
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=5, dilation=1),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, dilation=2),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU()
        )
        #self.avgpool = nn.AvgPool1d(kernel_size=10,stride=10,padding=2)

        if input_size==204:
            self.fc1 = nn.Linear(in_features=400,out_features=1000)#900
        else:
            self.fc1 = nn.Linear(in_features=700, out_features=1000)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=1000, out_features=1)

    def forward(self, inputs):
        #print(inputs.shape)
        # if self.input_size == 204:
        #     x = inputs[:,None,:]#2,1,205
        # else:
        #     x = img
        x = self.avgpool(inputs)
        #print(x.shape)# 2,1,20
       # x = torch.concat([x,indice],dim=-1)
        x = self.bn0(x)
        x = self.convblock1(x)
        #np.save('feature_map3.npy', x.detach().numpy())
        #print(x.shape)
        x = self.convblock2(x)# b,c,t
        #print(x.shape)

       # # print(x.shape)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        #print(x.shape)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #print(x.shape)
        return x
#---------------------------仅用高光谱数据  CNN------------------------------



if __name__ == "__main__":
    #from torchsummary import summary

    x = torch.randn([1,1,204])
   
    net = OnedCNN(None, 204)
    out = net(x)
    #summary(net, input_size=(1, 1, 204))
    from ptflops import get_model_complexity_info
    import re
    macs,params = get_model_complexity_info(net,tuple(x.shape[1:]),as_strings=True,
                                            print_per_layer_stat=False,verbose=True)

    flops = eval(re.findall(r'([\d.]+)',macs)[0])*2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)',macs)[0][0]

    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))
