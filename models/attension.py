import torch
import torch.nn as nn
import math


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = 16 #num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Average pooling along each channel
        squeeze_tensor = input_tensor.mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelTimeSenseSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_in_channels, reduction_ratio=2, kersize=[3, 5, 10], subband_num=1):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelTimeSenseSELayer, self).__init__()
        num_channels = 16#num_channels // reduction_ratio
        #num_channels = num_in_channels #* 2 if num_in_channels > 4 else num_in_channels * 4
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_in_channels, num_in_channels, kernel_size=kersize[0], groups=num_in_channels // subband_num),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_in_channels, num_in_channels, kernel_size=kersize[1], groups=num_in_channels // subband_num),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_in_channels, num_in_channels, kernel_size=kersize[2], groups=num_in_channels // subband_num),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_in_channels, num_channels, bias=True)
        self.fc2 = nn.Linear(num_channels, num_in_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Extracting multi-scale information in the time dimension
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  # [B, num_channels, 3]
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  # [B, num_channels]

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b,_ = input_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1)) #B,C,1
        return output_tensor

#--------------------------------和上一个的区别：返回权重----------------------------------
# class ChannelTimeSenseSEWeightLayer(nn.Module):
#     """
#     Re-implementation of Squeeze-and-Excitation (SE) block described in:
#         *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
#     """
#
#     def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
#         """
#         :param num_channels: No of input channels
#         :param reduction_ratio: By how much should the num_channels should be reduced
#         """
#         super(ChannelTimeSenseSEWeightLayer, self).__init__()
#         num_channels_reduced = num_channels // reduction_ratio
#         self.reduction_ratio = reduction_ratio
#         self.smallConv1d = nn.Sequential(
#             nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels),  # [B, num_channels, T]
#             nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
#             nn.ReLU(inplace=True)
#         )
#         self.middleConv1d = nn.Sequential(
#             nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels),  # [B, num_channels, T]
#             nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
#             nn.ReLU(inplace=True)
#         )
#         self.largeConv1d = nn.Sequential(
#             nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels),  # [B, num_channels, T]
#             nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
#             nn.ReLU(inplace=True)
#         )
#         self.feature_concate_fc = nn.Linear(3, 1, bias=True)
#         self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
#         self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input_tensor):
#         """
#         :param input_tensor: X, shape = (batch_size, num_channels, T)
#         :return: output tensor
#         """
#         # batch_size, num_channels, T = input_tensor.size()
#         # Extracting multi-scale information in the time dimension
#         small_feature = self.smallConv1d(input_tensor)
#         middle_feature = self.middleConv1d(input_tensor)
#         large_feature = self.largeConv1d(input_tensor)
#
#         feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  # [B, num_channels, 3]
#         squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  # [B, num_channels]
#
#         # channel excitation
#         fc_out_1 = self.relu(self.fc1(squeeze_tensor))
#         fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
#
#         a, b = squeeze_tensor.size()
#         output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
#         return output_tensor, fc_out_2.view(a, b, 1)

#----------------------------------------两层卷积-------------------------------------------
class ChannelDeepTimeSenseSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelDeepTimeSenseSELayer, self).__init__()
        num_channels_reduced = 16 #num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels),  # [B, num_channels, T]
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # [B, num_channels, 1]
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels),  # [B, num_channels, T]
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels),  # [B, num_channels, T]
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # [B, num_channels, 1]
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels),  # [B, num_channels, T]
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels),  # [B, num_channels, T]
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # [B, num_channels, 1]

        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Extracting multi-scale information in the time dimension
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  # [B, num_channels, 3]
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  # [B, num_channels]

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))#[0.9,0.1,0,.0]

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor,fc_out_2


# class ChannelDeepTimeSenseSELayerV2(nn.Module):
#     """
#     Re-implementation of Squeeze-and-Excitation (SE) block described in:
#         *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
#     """
#
#     def __init__(self, num_in_channels, reduction_ratio=2, kersize=[3, 5, 10]):
#         """
#         :param num_channels: No of input channels
#         :param reduction_ratio: By how much should the num_channels should be reduced
#         """
#         super(ChannelDeepTimeSenseSELayerV2, self).__init__()
#         num_channels = num_in_channels * 2 if num_in_channels > 4 else num_in_channels * 4 #num_channels // reduction_ratio
#         self.reduction_ratio = reduction_ratio
#         self.smallConv1d = nn.Sequential(
#             nn.Conv1d(num_in_channels, num_channels, kernel_size=kersize[0], groups=num_in_channels),  # [B, num_channels, T]
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool1d(1)  # [B, num_channels, 1]
#         )
#         self.middleConv1d = nn.Sequential(
#             nn.Conv1d(num_in_channels, num_channels, kernel_size=kersize[1], groups=num_in_channels),  # [B, num_channels, T]
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels),  # [B, num_channels, T]
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool1d(1)  # [B, num_channels, 1]
#         )
#         self.largeConv1d = nn.Sequential(
#             nn.Conv1d(num_in_channels, num_channels, kernel_size=kersize[2], groups=num_in_channels),  # [B, num_channels, T]
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels),  # [B, num_channels, T]
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool1d(1)  # [B, num_channels, 1]
#
#         )
#         self.feature_concate_fc = nn.Linear(3, 1, bias=True)
#         self.fc1 = nn.Linear(num_channels, num_channels, bias=True)
#         self.fc2 = nn.Linear(num_channels, num_in_channels, bias=True)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input_tensor):
#         """
#         :param input_tensor: X, shape = (batch_size, num_channels, T)
#         :return: output tensor
#         """
#         # batch_size, num_channels, T = input_tensor.size()
#         # Extracting multi-scale information in the time dimension
#         small_feature = self.smallConv1d(input_tensor)
#         middle_feature = self.middleConv1d(input_tensor)
#         large_feature = self.largeConv1d(input_tensor)
#
#         feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  # [B, num_channels, 3]
#         squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  # [B, num_channels]
#
#         # channel excitation
#         fc_out_1 = self.relu(self.fc1(squeeze_tensor))
#         fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
#        # print(input_tensor.shape,fc_out_2.shape)
#         a, b,_ = input_tensor.size()
#         output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
#         return output_tensor

#------------------------后面使用的模块----------------------------------
class Conv_Attention_Block(nn.Module):
    def __init__(
            self,
            num_channels,
            kersize=[3, 5, 10]
    ):
        """
        Args:
            num_channels: No of input channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.conv1d = nn.Conv1d(num_channels, num_channels, kernel_size=kersize, groups=num_channels)
        self.attention = SelfAttentionlayer(amp_dim=num_channels, att_dim=num_channels)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.active_funtion = nn.ReLU(inplace=True)

    def forward(self, input):
        input = (self.conv1d(input)).permute(0, 2, 1)  # [B, num_channels, T]  =>  [B, T, num_channels]
        #print(input.shape)
        input = self.attention(input, input, input)  # [B, T, num_channels]
        output = self.active_funtion(self.avgpool(input.permute(0, 2, 1)))  # [B, num_channels, 1]
        return output

#------------------------------------卷积换成convattention---------------------------
class ChannelTimeSenseAttentionSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelTimeSenseAttentionSELayer, self).__init__()
        num_channels_reduced = 16 #num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        self.smallConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[0])
        self.middleConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[1])
        self.largeConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[2])

        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Extracting multi-scale information in the time dimension
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  # [B, num_channels, 3]
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  # [B, num_channels]

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor

#-----------------------CBAM通道注意力----------------------------------
class ChannelCBAMLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelCBAMLayer, self).__init__()
        num_channels_reduced = 16 #num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Average pooling along each channel
        mean_squeeze_tensor = input_tensor.mean(dim=2)
        max_squeeze_tensor, _ = torch.max(input_tensor, dim=2)  # input_tensor.max(dim=2)
        # channel excitation
        mean_fc_out_1 = self.relu(self.fc1(mean_squeeze_tensor))
        max_fc_out_1 = self.relu(self.fc1(max_squeeze_tensor))
        fc_out_1 = mean_fc_out_1 + max_fc_out_1
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = mean_squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor

#-------------------------SE的全连接换成一维卷积------------------------
class ChannelECAlayer(nn.Module):
    """
     a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, num_channels, k_size=3):
        super(ChannelECAlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SelfAttentionlayer(nn.Module):
    """
    Easy self attention.
    """

    def __init__(self, amp_dim=257, att_dim=257):
        super(SelfAttentionlayer, self).__init__()
        self.d_k = amp_dim
        self.q_linear = nn.Linear(amp_dim, att_dim)
        self.k_linear = nn.Linear(amp_dim, att_dim)
        self.v_linear = nn.Linear(amp_dim, att_dim)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(att_dim, amp_dim)

    def forward(self, q, k, v):
        q = self.q_linear(q)  # [B, T, F]
        k = self.k_linear(k)
        v = self.v_linear(v)
        output = self.attention(q, k, v)
        output = self.out(output)
        return output  # [B, T, F]

    # def forward(self,x):
    #     q = self.q_linear(x)  # [B, T, F]
    #     k = self.k_linear(x)
    #     v = self.v_linear(x)
    #     output = self.attention(q, k, v)
    #     output = self.out(output)
    #     return output  # [B, T, F]

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.sigmoid(scores)
        output = torch.matmul(scores, v)
        return output

from functools import reduce
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 计算从向量C降维到 向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=in_channels,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))    #[batch_size,out_channels,H,W]
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]
        #print(U.size())
        s=self.global_pool(U)     # [batch_size,channel,1,1]
        #print(s.size())
        z=self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
       # print(z.size())
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        #print(a_b.size())
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]
        #print(a_b.size())
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        #print(a_b[0].size())
        #print(a_b[1].size())
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        return V    # [batch_size,out_channels,H,W]

# x = torch.Tensor(1,32,24,24)
# conv = SKConv(32,32,1,2,16,32)
#
# print(conv(x).size())

