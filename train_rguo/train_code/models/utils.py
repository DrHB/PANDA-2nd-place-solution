import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight,0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)

def reset_attention(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.reset_parameters()


@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See additional documentation for mish class.

    '''
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return mish(x)


def convert_relu_to_mish(module):
    mod=module
    if isinstance(module,nn.ReLU):
        mod=Mish()
    for name,child in module.named_children():
        mod.add_module(name,convert_relu_to_mish(child))
    return mod


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def smooth_l1_loss(input, target, delta=1.):
    t = torch.abs(input - target)
    return torch.where(t < delta, 0.5 * t ** 2, t * delta - (0.5 * delta ** 2))


def set_momentum(model,momentum):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.momentum=momentum
    return model