import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time

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
        self.gpu = False
        
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
        end_time = time.time()
        uniform_samples_tensor = torch.cuda.FloatTensor(template_tensor.shape).uniform_()
        if verbose:
            print ('random', time.time() - end_time)
            end_time = time.time()

        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        if verbose:
            print ('log', time.time() - end_time)
            end_time = time.time()
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = len(logits.shape) - 1
        end_time = time.time()

        if verbose:
            print ('gumble_sample', time.time() - end_time)
            end_time = time.time()


        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
        #else:
        #    gumble_trick_log_prob_samples = logits 

        if verbose:
            print ('gumble_trick_log_prob_samples', time.time() - end_time)
            end_time = time.time()

        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)

        if verbose:
            print ('soft_samples', time.time() - end_time)
            end_time = time.time()
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
        #logits = 8 * (logits - torch.mean(logits,dim=[1,2],keepdim=True)) / (torch.var(logits,dim=[1,2],keepdim=True)  +1e-5)
        #logits = F.normalize(logits,p=2,dim=-1) * np.sqrt(16)
        end_time = time.time()
        dim = len(logits.shape) - 1

        y = self.gumbel_softmax_sample(logits, temperature)

        if verbose:
            print ('gumbel_softmax_sample', time.time() - end_time)

        if hard:
            end_time = time.time()

            _, max_value_indexes = y.data.max(dim, keepdim=True)
#            y_hard = torch.zeros_like(logits).scatter_(1, max_value_indexes, 1)


            if verbose:
                print ('max_value_indexes', time.time() - end_time)
                end_time = time.time()

            y_hard = logits.data.clone().zero_().scatter_(dim, max_value_indexes, 1)


            if verbose:
                print ('y_hard', time.time() - end_time)
                end_time = time.time()

            y = Variable(y_hard - y.data) + y


            if verbose:
                print ('y', time.time() - end_time)
                end_time = time.time()
#            exit(1)

            if index:
                return idx
        return y
        
    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()
        force_hard = False
        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True) 

import torch.nn as nn
import math
class GumbleSoftmaxV2(torch.nn.Module):
    def __init__(self, num_classes=204):
        super(GumbleSoftmaxV2, self).__init__()
        self.n_claster =  1
        self.num_classes = num_classes
        self.s_int =  math.sqrt(2.0) * math.log(num_classes *  self.n_claster)  #7.4 -> 11.3
        self.W = nn.Parameter(torch.zeros([num_classes,num_classes * self.n_claster]),requires_grad=False)
        self.s = nn.Parameter(torch.tensor(self.s_int),requires_grad=False)
        nn.init.kaiming_uniform_(self.W)
        #self.theta =  False
        #self.eps = 1e-7
        self.softmax = GumbleSoftmax()
    def forward(self, logits, temp=1, force_hard=False):
       # temp = 0.5
        logits = F.normalize(logits,dim=-1,p=2)        
        W = F.normalize(self.W,dim=0,p=2)        
        logits = logits @ W
        #theta = torch.acos(torch.clip(logits,-1+self.eps,1-self.eps))
        b,c,t = logits.shape
        #print(self.s,logits.min(),logits.max())
        if self.training:
            max_s_logits = torch.max(self.s * logits)
            B_avg = torch.exp(self.s * logits - max_s_logits)
            B_avg = torch.mean(torch.sum(B_avg,dim=-1))

            #theta_class = torch.mean(theta.reshape(b,c,self.n_claster,self.num_classes),dim=-2)
            #theta_med = torch.quantile(theta_class,0.5)
            #print(B_avg,self.s)
            self.s.data = (max_s_logits + torch.log(B_avg)) / 0.707 # torch.cos(torch.clamp_max(theta_med,torch.pi/4.0))  #0.707
        logits *= self.s

        # if self.training and not force_hard:
        #     out =  F.gumbel_softmax(logits,tau=temp,hard=False)
        # else:
        #     out = F.gumbel_softmax(logits,tau=temp,hard=True)
        # #print(logits.min(),logits.max()) +- 2.5
        # out = self.softmax(logits,1,True)
        if self.training:
            out = gumbel_softmax_hand(logits, tau=temp, hard=True,  dim=-1,gumber=True)
        else:
            out = gumbel_softmax_hand(logits, tau=temp, hard=True,  dim=-1,gumber=False)

        logits = out.reshape(b,c,self.n_claster,self.num_classes)
        logits = torch.sum(logits,dim=-2) 
        return logits

class GumbleSoftmaxV3(torch.nn.Module):
    def __init__(self, num_classes=204):
        super(GumbleSoftmaxV3, self).__init__()
        self.n_claster =  1
        self.num_classes = num_classes
        # self.s_int =  math.sqrt(2.0) * math.log(num_classes *  self.n_claster)  #7.4 -> 11.3
        # self.W = nn.Parameter(torch.zeros([num_classes,num_classes * self.n_claster]),requires_grad=False)
        # self.s = nn.Parameter(torch.tensor(self.s_int),requires_grad=False)
        # nn.init.kaiming_uniform_(self.W)
        #self.theta =  False
        #self.eps = 1e-7
        self.softmax = GumbleSoftmax()
    def forward(self, logits, temp=1, force_hard=False):
       # temp = 0.5
        logits = F.normalize(logits,dim=-1,p=2)        
        # W = F.normalize(self.W,dim=0,p=2)        
        # logits = logits @ W
        # #theta = torch.acos(torch.clip(logits,-1+self.eps,1-self.eps))
        b,c,t = logits.shape
        # #print(self.s,logits.min(),logits.max())
        # if self.training:
        #     max_s_logits = torch.max(self.s * logits)
        #     B_avg = torch.exp(self.s * logits - max_s_logits)
        #     B_avg = torch.mean(torch.sum(B_avg,dim=-1))

        #     #theta_class = torch.mean(theta.reshape(b,c,self.n_claster,self.num_classes),dim=-2)
        #     #theta_med = torch.quantile(theta_class,0.5)
        #     #print(B_avg,self.s)
        #     self.s.data = (max_s_logits + torch.log(B_avg)) / 0.707 # torch.cos(torch.clamp_max(theta_med,torch.pi/4.0))  #0.707
        # logits *= self.s

        # if self.training and not force_hard:
        #     out =  F.gumbel_softmax(logits,tau=temp,hard=False)
        # else:
        #     out = F.gumbel_softmax(logits,tau=temp,hard=True)
        # #print(logits.min(),logits.max()) +- 2.5
        # out = self.softmax(logits,1,True)
        if self.training:
            out = gumbel_softmax_hand(logits, tau=temp, hard=True,  dim=-1,gumber=True)
        else:
            out = gumbel_softmax_hand(logits, tau=temp, hard=True,  dim=-1,gumber=False)

        logits = out.reshape(b,c,self.n_claster,self.num_classes)
        logits = torch.sum(logits,dim=-2) 
        return logits


def gumbel_softmax_hand(logits, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1,gumber=True) :

    if gumber:
        gumbels =  - (torch.rand_like(logits) / 2.0 +1e-7)  #1.4  0.5 < 1.4/2

        # uniform_samples_tensor = torch.FloatTensor(logits.shape).uniform_().to(logits.device)
        # gumbels = - torch.log(eps - torch.log(uniform_samples_tensor + eps))  / 10.0
        
    else:
        gumbels = 0

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
