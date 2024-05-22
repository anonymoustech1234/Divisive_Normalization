# OUR DIVNORM FUNC
# BY HEEGER'S FORMULA

import torch
import torch.nn as nn
from torch.nn import functional as F

# OUR DIVNORM FUNC
# BY HEEGER"S FORMULA
import torch
import torch.nn as nn
from torch.nn import functional as F

import math


class DivisiveNormHeeger(nn.Module):
    r"""Div norm from Heeger's divisive normalization formula

    math: b_c(x)=\frac{\gamma a_c(x)^2}{\left(\sigma+\frac{\alpha}{\lambda} \sum a_{c+j}(x)^2 \right)}

    """
    def __init__(self, output_channels=64, args_divnorm=None):
        super(DivisiveNormHeeger, self).__init__()
        self.args_divnorm = args_divnorm
        self.power = args_divnorm['power']

        gamma = args_divnorm['gamma']
        sigma = args_divnorm['sigma']
        p = args_divnorm['p']

        pool_size = 2**int(p * math.log2(output_channels))

        #neighborhood_size = number of neurons in each pool
        #num_neighbors = number of pools

        self.num_pools = output_channels // pool_size
        self.pool_size = pool_size

        if args_divnorm['fix_all'] == True:
            args_divnorm['fix_gamma'] = True
            args_divnorm['fix_sigma'] = True

        if args_divnorm['single_gamma']:
            self.gamma = nn.Parameter(torch.Tensor([gamma]))
            if args_divnorm['fix_gamma']: # fix needs single first
                self.gamma.requires_grad = False
        else:
            self.gamma = nn.Parameter(torch.Tensor([gamma]*(self.num_pools))) #one gamma per pool
        if args_divnorm['single_sigma']:
            self.sigma = nn.Parameter(torch.Tensor([sigma])) # bias = sigma
            if args_divnorm['fix_sigma']:
                self.sigma.requires_grad = False
        else:
            self.sigma = nn.Parameter(torch.Tensor([sigma]*(self.num_pools)))

        

    def forward(self, input):
        return divisive_normalization_Heeger(
            input, gamma=self.gamma, sigma=self.sigma, power=self.power, neighborhood_size=self.pool_size, num_neighbors=self.num_pools, args=self.args_divnorm
        )

    def extra_repr(self):
        return f'gamma={self.gamma},sigma={self.sigma}, neighborhood_size={self.pool_size}'
    
def divisive_normalization_Heeger(input, gamma=torch.ones(8), sigma=torch.zeros(8), power=2, neighborhood_size=8, num_neighbors=8, args=None):
    """
    Expect gamma, sigma to be 1D tensor . shape = nerighbornum
    """
    input_shape = input.shape
    if "eps" in args:
        eps = args['eps']
    else:
        eps = 1e-8
    #check input dimension
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    
    if power % 2:
        x = torch.abs(x)

    #sqaure the input    
    #x = input.mul(input).unsqueeze(1) 
    input=input.pow(power).unsqueeze(1) 

    # code commented out while debugging NVIDIA DALI Loader
    # device = input.get_device()
    # weits = torch.ones(neighborhood_size).reshape(1,1,neighborhood_size,1,1)
    # if device != -1:
    #     weits = weits.to(device = device)

    #initialize filter weight (for summation in denominator: {\Sigma a^2_j})
    
    weits = torch.ones(neighborhood_size).reshape(1,1,neighborhood_size,1,1)
    if torch.cuda.is_available():
        weits = weits.to(torch.device('cuda'))
    div = F.conv3d(input, weits, stride = (neighborhood_size,1,1)) 
    # print(weits.shape)

    if args['single_sigma']:
        num = (div + (sigma.expand(1, 1, num_neighbors, 1, 1) ** power) + eps)
    else:
        if args['normalize_sigma']:
            sigma = sigma**power / torch.sum(sigma**power)
        num = (div + (sigma.reshape(1, 1, num_neighbors, 1, 1) ** power) + eps)
    if args['single_gamma']:
        den = torch.exp(gamma).expand(1, 1, num_neighbors, 1, 1)
    else:
        if args['normalize_gamma']:
            gamma = gamma**power / torch.sum(sigma**power)
        den = torch.exp(gamma).reshape(1, 1, num_neighbors, 1, 1)

    din = num / den
    din_exp=din.permute(0,2,1,3,4).expand(-1, -1, neighborhood_size, -1, -1).reshape(input_shape)
    
    return input.squeeze() / din_exp
