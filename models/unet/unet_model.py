""" Full assembly of the parts to form the complete network """

from .unet_parts import *
#from models.TCN import TCNBlock
#https://github.com/milesial/Pytorch-UNet

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, encs=[64,128,256,512],hidden=16, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, encs[0]))
        self.down1 = (Down(encs[0], encs[1]))
        self.down2 = (Down(encs[1], encs[2]))
        self.down3 = (Down(encs[2], encs[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(encs[3], encs[3]))
        self.compres = nn.Linear(encs[3],hidden)
        self.lstm = nn.LSTM(hidden,hidden,batch_first=True)
        self.decompres = nn.Linear(hidden,encs[3])#  64 vcmax

        # self.lstm = nn.LSTM(12,12,batch_first=True)

        #self.attention = nn.TransformerEncoderLayer(d_model=20,nhead=5,dim_feedforward=5120,batch_first=True)
        #self.tcn = TCNBlock(in_channels=20, hidden_channel=100, out_channels=20, dilation=2)
        self.up1 = (Up(encs[3], encs[2] , bilinear))
        self.up2 = (Up(encs[2], encs[1] , bilinear))
        self.up3 = (Up(encs[1], encs[0] , bilinear))
        self.up4 = (Up(encs[0], encs[0], bilinear))
        self.outc = (OutConv(encs[0], n_classes)) # 204  1  vcmax

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)

        #print(x5.shape)#7,,512,20
        x5 = x5.transpose(1,2)
        x5 = self.compres(x5)
        x5,h = self.lstm(x5) # B,T,C
        x5 = self.decompres(x5)
        x5 = x5.transpose(1,2)

        #x5 = self.attention(x5)

        # x5 = x5.transpose(1,2)
        # x5 = self.tcn(x5)
        # x5 = x5.transpose(1, 2)

        x = self.up1(x5, x4)
        #print(x.shape,x3.shape)
        x = self.up2(x, x3)
        #print(x.shape)

        x = self.up3(x, x2)
        #print(x.shape)

        x = self.up4(x, x1)
        #print(x.shape)

        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)