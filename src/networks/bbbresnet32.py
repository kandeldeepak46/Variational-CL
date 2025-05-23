import torch
import torch.nn as nn
from layers.BBBConv import BBBConv2d, BBBConv2D  # Make sure this is correctly imported based on your setup
from layers.BBBLinear import BBBLinear

__all__ = ['bbbresnet32']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 Bayesian convolution with padding."""
    return BBBConv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return BBBConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class NoReLUBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(NoReLUBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_features=64):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1_starting = BBBConv2D(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_starting = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(NoReLUBasicBlock, 64, layers[2], stride=2)
        # last classifier layer (head) with as many outputs as classes
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = BBBLinear(64 * block.expansion, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BBBConv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1_starting(self.conv1_starting(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = torch.mean(x, dim=(2, 3))
        # x = nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12)
        x = self.fc(features)
        if return_features:
            return x, features
        return x
    
    def KL(self):
        kl_loss = 0
        for module in self.modules():
            if isinstance(module, BBBConv2D):
                kl_loss += module.kl_loss()
        for module in self.modules():
            if isinstance(module, BBBLinear):
                kl_loss += module.kl_loss()   
        return kl_loss


def bbbresnet32(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    # change n=3 for ResNet-20, and n=9 for ResNet-56
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model
