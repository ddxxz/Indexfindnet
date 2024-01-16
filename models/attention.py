import torch
import torch.nn as nn

# z = [ 0,2,4,6 ]        #
# i = 0 : e^0 /  ( e^0 + e^2 + e^4 + e^6 )
# i = 1 : e^2 /  ( e^0 + e^2 + e^4 + e^6 )

class Attention(nn.Module):
    def __init__(self, ):
        super(Attention, self).__init__()
        self.Wq = nn.Linear(128, 128,bias=False)
        self.Wk = nn.Linear(128, 128, bias=False)
        self.Wv = nn.Linear(128, 128, bias=False)
    def forward(self, inputs):
        print(inputs.shape)#B , T , C
        Q = self.Wq(inputs) #（4，128）*（128，128）=（4，128)
        K = self.Wk(inputs) #（4，100,128）*（128，128）=（4，100,128)
        V = self.Wv(inputs)
        A=torch.matmul(K.transpose(1,2),Q) #(128,4)*(4,128)=(128,128)
                            # e-xi /sum(0,n) e^(-xi)
        A_=torch.softmax(A,dim=-1)
        O=torch.matmul(V,A_) #
        #O = V * A_
        return O

# x = torch.randn([4,10,128])# T
# att = Attention()
# q = att(x)
# print(q.shape)
#
layer1 = nn.TransformerEncoderLayer(d_model=50, nhead=5, dim_feedforward=50*4,
                                    batch_first=True)

#q = layer1(x)
# print(q.shape)

x = torch.randn([16,16,8])
layer = nn.Conv1d(16,50,3,padding=1)
r = layer(x)
print(r.shape)
r = r .transpose(1,2)
r2 = layer1(r)
print(r2.shape)