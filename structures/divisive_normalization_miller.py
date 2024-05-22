from __future__ import division
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch._C import _infer_size
from torch._C import *
import math

class DivisiveNormKenMiller(nn.Module):
    r"""Applies divisive normalization along the feature dimension with an exponential kernel
    math: b_c(x)=\frac{a_c(x)^2}{\left(k\left(1+\frac{\alpha}{\lambda} \sum_{j=-4 \lambda}^{4 \lambda} a_{c+j}(x)^2 e^{-|j| / \lambda}\right)\right)^\beta}
    """

    def __init__(self, args_divnorm):
        super(DivisiveNormKenMiller, self).__init__()
        self.lamb = Parameter(torch.Tensor([args_divnorm['lamb']]))
        self.alpha = Parameter(torch.Tensor([args_divnorm['alpha']]))
        self.beta = Parameter(torch.Tensor([args_divnorm['beta']]))
        self.k = Parameter(torch.Tensor([args_divnorm['k']]))

        if args_divnorm['fix_all']:
            self.lamb.requires_grad = False
            self.alpha.requires_grad = False
            self.beta.requires_grad = False
            self.k.requires_grad = False
        

    def forward(self, input):
    
        return divisive_normalization_kenmiller(input, self.lamb, self.alpha, self.beta, self.k)

    def extra_repr(self):
        return 'lambda={lamb},alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__['_parameters'])


def divisive_normalization_kenmiller(input, lamb=5, alpha=5, beta=0.75, k=1.):
    """
    Applies normalization across channels with exponential kernel
    """
    device_a = lamb.device

    if math.isnan(lamb.item()):
        lamb = torch.Tensor([5]).to(device_a)
    if math.isnan(alpha.item()):
        alpha = torch.Tensor([0.1]).to(device_a)
    if math.isnan(beta.item()):
        beta = torch.Tensor([1]).to(device_a)
    if math.isnan(k.item()):
        k = torch.Tensor([1]).to(device_a)
    neighbors = int(torch.ceil(2 * 4 * lamb).item())
    if neighbors % 2 == 0:
        neighbors = neighbors + 1
    else:
        pass
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)
    # hacky trick to try and keep everything on cuda
    sizes = input.size()
    weits = input.clone().detach()
    weits = weits.new_zeros(([1] + [1] + [int(neighbors)] + [1] + [1]))

    if dim == 3:
        div = F.pad(div, (0, 0, neighbors // 2, neighbors - 1 // 2))
        div = torch._C._nn.avg_pool2d((div, neighbors, 1), stride=1).squeeze(1)
    else:
        dev = input.get_device()
        # indexx is a 1D tensor that is a symmetric exponential distribution of some "radius" neighbors
        idxs = torch.abs(torch.arange(neighbors) - neighbors // 2)
        weits[0, 0, :, 0, 0] = idxs
        weits = torch.exp(-weits / lamb)
        # creating single dimension at 1;corresponds to number of input channels;only 1 input channel
        # 3D convolution; weits has dims: Cx1xCx1x1 ; this means we have C filters for the C channels
        # The div is the input**2; it has dimensions B x 1 x C x W x H
        div = F.conv3d(div, weits, padding=((neighbors // 2), 0, 0))
        div = div / lamb

    div = div.mul(alpha).add(1).mul(k).pow(beta)
    div = div.squeeze()
    return input.mul(input) / div
