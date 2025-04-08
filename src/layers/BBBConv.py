import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .misc import KL_DIV, ModuleWrapper


class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):

        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.001,
                'posterior_mu_initial': (0, 0.001),
                'posterior_rho_initial': (-3, 0.001),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
    

class BBBConv2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, priors=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.0001,
            }
        self.prior_W_mu = priors['prior_mu']
        self.prior_W_sigma = priors['prior_sigma']
        self.prior_bias_mu = priors['prior_mu']
        self.prior_bias_sigma = priors['prior_sigma']

        # Define nn.Conv2d layers for weights
        self.conv_mu = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.conv_rho = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

        # Initialize rho parameters (used for log(sigma))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize mean weights and biases
        self.conv_mu.weight.data.normal_(0, 0.1)
        if self.use_bias:
            self.conv_mu.bias.data.normal_(0, 0.1)

        # Initialize rho to control sigma (log-space)
        self.conv_rho.weight.data.fill_(-3)
        if self.use_bias:
            self.conv_rho.bias.data.fill_(-3)

    def forward(self, x, sample=True):
        # Compute sigma from rho using softplus approximation
        W_sigma = torch.log1p(torch.exp(self.conv_rho.weight))
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.conv_rho.bias))
            bias_var = bias_sigma ** 2
        else:
            bias_sigma = bias_var = None

        act_mu = self.conv_mu(x)
        act_var = 1e-16 + F.conv2d(x ** 2, W_sigma ** 2, bias=bias_var, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty_like(act_mu).normal_(0, 0.0001).to(act_mu.device)
            return act_mu   + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = KL_DIV(self.prior_W_mu, self.prior_W_sigma, self.conv_mu.weight, 
                    torch.log1p(torch.exp(self.conv_rho.weight)))
        if self.use_bias:
            kl += KL_DIV(self.prior_bias_mu, self.prior_bias_sigma, self.conv_mu.bias, 
                         torch.log1p(torch.exp(self.conv_rho.bias)))
        return kl
