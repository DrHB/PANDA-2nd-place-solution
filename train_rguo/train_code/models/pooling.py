import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import GeM

class AttentionPool(nn.Module):
    def __init__(self,in_ch,hidden=512,dropout=0.25):
        super().__init__()
        self.in_ch=in_ch

        module=[nn.Linear(in_ch,hidden,bias=True),
                nn.Tanh()
                ]
        if dropout>0:
            module.append(nn.Dropout(dropout))
        module.append(nn.Linear(hidden,1,bias=True))
        self.attention=nn.Sequential(*module)

    def forward(self,x):
        num_patch=x.size(1)
        x=x.view(-1,x.size(2))
        A=self.attention(x)
        A=A.view(-1,num_patch,1)
        wt=F.softmax(A,dim=1)
        return (x.view(-1,num_patch,self.in_ch)*wt).sum(dim=1),A


class AttentionPool_Gated(nn.Module):
    def __init__(self,in_ch,hidden=512,dropout=0.25):
        super().__init__()
        self.in_ch=in_ch

        module_a=[nn.Linear(in_ch,hidden,bias=True),
                nn.Tanh()
                ]
        module_b=[nn.Linear(in_ch,hidden,bias=True),
                nn.Sigmoid()
                ]
        if dropout>0:
            module_a.append(nn.Dropout(dropout))
            module_b.append(nn.Dropout(dropout))
        self.attention_a=nn.Sequential(*module_a)
        self.attention_b=nn.Sequential(*module_b)
        self.fc=nn.Linear(hidden,1,bias=True)
    def forward(self,x):
        num_patch=x.size(1)
        x=x.view(-1,x.size(2))
        a=self.attention_a(x)
        b=self.attention_b(x)
        A=self.fc(a*b)
        A=A.view(-1,num_patch,1)
        wt=F.softmax(A,dim=1)
        return (x.view(-1,num_patch,self.in_ch)*wt).sum(dim=1),A


class AdaptiveConcatPool2d_Attention(nn.Module):
    def __init__(self, in_ch,hidden=512,dropout=0.25,gated=False):
        super().__init__()
        sz = (1,1)

        if gated:
            self.ap=AttentionPool_Gated(in_ch,hidden,dropout)
        else:
            self.ap = AttentionPool(in_ch,hidden=hidden,dropout=dropout)
        self.mp = nn.AdaptiveMaxPool2d(sz)
        self.in_ch=in_ch
    def forward(self, x):
        ap,A=self.ap(x)#[batch,num_patch,C]
        mp=torch.max(x,dim=1)[0]
        return torch.cat([ap, mp], dim=1),A

