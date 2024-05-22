# OUR DIVNORM FUNC
# BY HEEGER'S FORMULA
# WHEN ONLY ONE POOL

import torch
import torch.nn as nn
from torch.nn import functional as F

import math

class DivNorm2d(nn.Module):
    def __init__(self, args_divnorm=None):
        super(DivNorm2d, self).__init__()
        self.gamma = nn.Parameter(torch.tensor([args_divnorm['gamma']]))
        self.sigma = nn.Parameter(torch.tensor([args_divnorm['sigma']]))
        self.eps = args_divnorm['eps']
        self.power = args_divnorm['power']

        if args_divnorm['fix_all']:
            args_divnorm['fix_gamma'] = True
            args_divnorm['fix_sigma'] = True

        if args_divnorm['fix_gamma']:
            self.gamma.requires_grad = False
        if args_divnorm['fix_sigma']:
            self.sigma.requires_grad = False

    def forward(self, x):
        if self.power % 2:
            x = torch.abs(x)
        x = x.pow(self.power)
        total_sum = torch.sum(x, dim=1, keepdim=True)
        return torch.exp(self.gamma[0]) * x / (self.sigma[0] ** self.power + total_sum + self.eps)