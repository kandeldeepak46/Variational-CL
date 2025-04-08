from torch import nn
import torch.nn.functional as F


from layers.BBBLinear import BBBLinear
from layers.BBBConv import BBBConv2d, BBBConv2D  


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 Bayesian convolution with padding."""
    return BBBConv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return BBBConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LeNetArch(nn.Module):
    """LeNet-like network for tests with MNIST (28x28)."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()
        # main part of the network
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(84, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc(out)
        return out


def LeNet(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    return LeNetArch(**kwargs)

class BayesianLeNetArch(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, **kwargs):
        super().__init__()
        # main part of the network
        self.conv1 = BBBConv2D(in_channels, 6, 5)
        self.conv2 = BBBConv2D(6, 16, 5)
        self.fc1 = BBBLinear(16 * 16, 120)   # for MNIST
        self.fc1 = BBBLinear(16 * 25, 120)   # for CIFAR10
        self.fc2 = BBBLinear(120, 64)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(64, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x, return_features=False):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))  
        features = out
        if return_features:
            return features, self.fc(out)
        return self.fc(out)
    
    def get_kl_loss(self):
        kl_loss = 0
        for module in self.modules():
            if isinstance(module, BBBConv2d):
                kl_loss += module.kl_loss()
        for module in self.modules():
            if isinstance(module, BBBLinear):
                kl_loss += module.kl_loss()
        return kl_loss
    

def BayesianLeNet(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    return BayesianLeNetArch(**kwargs)
