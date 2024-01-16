import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import math

"""
Gumbel Softmax Sampler
Requires 2D input [batchsize, number of categories]

Does not support sinlge binary category. Use two dimensions with softmax instead.
"""

verbose = False

class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard
        self.gpu = True#False
        
    def cuda(self):
        self.gpu = True
    
    def cpu(self):
        self.gpu = False
        
    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        #noise = torch.cuda.FloatTensor(shape).uniform_()
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise)

#        noise = torch.rand(shape)
#        noise.add_(eps).log_().neg_()
#        noise.add_(eps).log_().neg_()
#        if self.gpu:
#            return Variable(noise).cuda()
#        else:
#            return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        #end_time = time.time()
        uniform_samples_tensor = torch.cuda.FloatTensor(template_tensor.shape).uniform_()
        #uniform_samples_tensor = torch.cuda.FloatTensor(*template_tensor.shape).uniform_()
        #print('tem',template_tensor.shape)
        #uniform_samples_tensor = torch.tensor(template_tensor,dtype=torch.float32,device='cuda').uniform_()
        #print('uniform_samples_tensor',uniform_samples_tensor)
        # uniform_samples_tensor = torch.FloatTensor(template_tensor.shape).uniform_()
        # uniform_samples_tensor = uniform_samples_tensor.to('cuda')
        # if verbose:
        #     print ('random', time.time() - end_time)
        #     end_time = time.time()

        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        # if verbose:
        #     print ('log', time.time() - end_time)
        #     end_time = time.time()
        return gumble_samples_tensor.to('cuda')

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""

        #x = torch.randn(2, 3, 4)
        # 定义需要加的数
        value = 20000
        # 将 value 转化为一个和 x 形状相同的张量
        value_tensor = torch.tensor(value, dtype=logits.dtype, device=logits.device)
        value_tensor = value_tensor.expand_as(logits)
        # 将 x 和 value_tensor 相加
        logits = logits + value_tensor

        dim = len(logits.shape) - 1
        #end_time = time.time()


        # if verbose:
        #     print ('gumble_sample', time.time() - end_time)
        #     end_time = time.time()

        #if self.training:
        #print('data',logits.data.shape)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        #print('shape',torch.max(logits),torch.min(logits),torch.max(Variable(gumble_samples_tensor)),torch.min(Variable(gumble_samples_tensor)))
        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
        #print(torch.cuda.current_device())
        #else:
        #    gumble_trick_log_prob_samples = logits 

        # if verbose:
        #     print ('gumble_trick_log_prob_samples', time.time() - end_time)
        #     end_time = time.time()

        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)

        # if verbose:
        #     print ('soft_samples', time.time() - end_time)
        #     end_time = time.time()
        return soft_samples
    
    def gumbel_softmax(self, logits, temperature, hard=False, index=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [ ..., n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [..., n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        #print(logits.shape,logits.max(dim=-1)[0])
        # if logits.shape[-1] == 3:
        #     print(logits.max(dim=-1))

        #logits = F.normalize(logits,p=2,dim=-1) * np.sqrt(16)
        #end_time = time.time()
        logits = logits.to('cuda')
        dim = len(logits.shape) - 1

        y = self.gumbel_softmax_sample(logits, temperature)

        if verbose:
            pass#print ('gumbel_softmax_sample', time.time() - end_time)

        if hard:
            #end_time = time.time()

            _, max_value_indexes = y.data.max(dim, keepdim=True)
#            y_hard = torch.zeros_like(logits).scatter_(1, max_value_indexes, 1)


            if verbose:
                pass
                #print ('max_value_indexes', time.time() - end_time)
                #end_time = time.time()

            y_hard = logits.data.clone().zero_().scatter_(dim, max_value_indexes, 1)


            if verbose:
                pass
                #print ('y_hard', time.time() - end_time)
                #end_time = time.time()

            y = Variable(y_hard - y.data) + y


            if verbose:
                pass
                #print ('y', time.time() - end_time)
                #end_time = time.time()
#            exit(1)

            if index:
                return idx
        return y
        
    def forward(self, logits, temp=1, force_hard=False):
        #print('logits',torch.max(logits),torch.min(logits))
        samplesize = logits.size()

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True) 

class GumbelSoftmaxV3(torch.nn.Module):
    def __init__(self,num_classes):
        super(GumbelSoftmaxV3,self).__init__()
        self.n_claster = 1
        self.num_classes =num_classes
        self.s_int = math.sqrt(2.0)*math.log(num_classes*self.n_claster)
        self.W = nn.Parameter(torch.zeros([num_classes,num_classes*self.n_claster]),requires_grad=False)
        self.s = nn.Parameter(torch.tensor(self.s_int),requires_grad=False)
        nn.init.kaiming_uniform_(self.W)
    def forward(self,logits,temp=1,force_hard=False):#0.001 20
        logits = F.normalize(logits,dim=-1,p=2)# x / |x|  #p=2 表示使用 L2 范数来计算向量的模。F.normalize(logits, dim=-1, p=2)
        W = F.normalize(self.W,dim=0,p=2)      # w / |w|
        logits = logits @ W  #-1,1  -0.2 0.2   -0.6 0.6                # x * w/(|x||w|)  cos
        #theta = torch.acos(logits)
        #print(logits.min(),logits.max(),self.s)

        if self.training:
            max_s_logits = torch.max(self.s*logits)
            B_avg = torch.exp(self.s*logits-max_s_logits)
            B_avg = torch.mean(torch.sum(B_avg,dim=-1))
            self.s.data = (max_s_logits+torch.log(B_avg))/0.707
        logits *= self.s   #0.4 * 6  2.4

        out = Gumbel_Softmax(v=logits,t=temp,hard=True,training=self.training)

        b,c,t = logits.shape
        logits = out.reshape(b,c,self.n_claster,self.num_classes)
        logits = torch.sum(logits,dim=-2)
        return logits
    
def Gumbel_Softmax(v,t=0.5,dim=-1,hard=False,eps=1e-10,training=True):
    logits=v
    n = v.shape
    # 生成n个服从均匀分布U(0, 1)的独立样本
    #c = torch.rand(n)
    # if training:
    #     c = torch.FloatTensor(n).to(v.device).uniform_()
    #     gi=-torch.log(-torch.log(c+eps) + eps)# max 2
    #     gi = gi / 10.0 # max 0.2
    # else:
    #     gi = 0
    if training:
        gi =  - (torch.rand_like(logits)/10.0+1e-7)  

        # uniform_samples_tensor = torch.FloatTensor(logits.shape).uniform_().to(logits.device)
        # gumbels = - torch.log(eps - torch.log(uniform_samples_tensor + eps))  / 10.0
    else:
        gi = 0

    v_=v+gi 
    y_soft = torch.softmax(v_/t,dim=dim)  #torch.exp((torch.log(pi)+gi)/ti)/torch.sum(torch.exp((torch.log(pk)+gk)/tk))
    
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret        

class GumbelSoftmaxV2(torch.nn.Module):
    def __init__(self,num_classes):
        super(GumbelSoftmaxV2,self).__init__()
        self.n_claster = 1
        self.num_classes =num_classes
        self.s_int = math.sqrt(2.0)*math.log(num_classes*self.n_claster)
        self.W = nn.Parameter(torch.zeros([num_classes,num_classes*self.n_claster]),requires_grad=False)
        self.s = nn.Parameter(torch.tensor(self.s_int),requires_grad=False)
        nn.init.kaiming_uniform_(self.W)
    def forward(self,logits,temp=1,force_hard=False):#0.001 20
        logits = F.normalize(logits,dim=-1,p=2)# x / |x|
        W = F.normalize(self.W,dim=0,p=2)      # w / |w|
        logits = logits @ W  #-1,1  -0.2 0.2   -0.6 0.6                # x * w/(|x||w|)  cos
        print(logits.min(),logits.max(),self.s)

        if self.training:
            max_s_logits = torch.max(self.s*logits)
            B_avg = torch.exp(self.s*logits-max_s_logits)
            B_avg = torch.mean(torch.sum(B_avg,dim=-1))
            self.s.data = (max_s_logits+torch.log(B_avg))/0.707
        logits *= self.s   #0.4 * 6  2.4
         
        if self.training and not force_hard:
            out = F.gumbel_softmax(logits,tau=temp,hard=False)
        else:
            out = F.gumbel_softmax(logits,tau=temp,hard=True)
        b,c,t = logits.shape
        logits = out.reshape(b,c,self.n_claster,self.num_classes)
        logits = torch.sum(logits,dim=-2)
        return logits




