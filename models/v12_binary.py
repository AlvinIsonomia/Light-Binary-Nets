import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
import numpy as np
import os

__all__ = ['v12_binary']


class MobileNet(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#                 BinarizeConv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.Hardtanh(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#                 BinarizeConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Hardtanh(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#                 BinarizeConv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.Hardtanh(inplace=True),
            )
        def bconv_dw(inp,oup,stride):
            return nn.Sequential(
                BinarizeConv2d(inp,inp,3,stride,1,groups = inp ,bias = True),
                nn.BatchNorm2d(inp),
                nn.Hardtanh(inplace = True),
                BinarizeConv2d(inp,oup,1,1,0,bias = True),
                nn.BatchNorm2d(oup),
                nn.Hardtanh(inplace=True),
            )
        
        self.ratioInfl = 2
        self.model = nn.Sequential(
            conv_bn(  3,  32*self.ratioInfl, 2), 
            conv_dw( 32*self.ratioInfl,  64*self.ratioInfl, 1),
            conv_dw( 64*self.ratioInfl, 128*self.ratioInfl, 2),
            conv_dw(128*self.ratioInfl, 128*self.ratioInfl, 1),
            conv_dw(128*self.ratioInfl, 256*self.ratioInfl, 2),
            bconv_dw(256*self.ratioInfl, 256*self.ratioInfl, 1),
            bconv_dw(256*self.ratioInfl, 512*self.ratioInfl, 2),
            bconv_dw(512*self.ratioInfl, 512*self.ratioInfl, 1),
#             conv_dw(512*self.ratioInfl, 512*self.ratioInfl, 1),
#             conv_dw(512*self.ratioInfl, 512*self.ratioInfl, 1),
#             conv_dw(512*self.ratioInfl, 512*self.ratioInfl, 1),
#             conv_dw(512*self.ratioInfl, 512*self.ratioInfl, 1),
            bconv_dw(512*self.ratioInfl, 1024*self.ratioInfl, 2),
            bconv_dw(1024*self.ratioInfl, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_classes)
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            25: {'lr': 1e-2},
            50: {'lr': 5e-4},
            75: {'lr': 1e-5},
            100: {'lr': 1e-6},
            150: {'lr': 1e-7},
            200: {'lr': 1e-6},
            250: {'lr': 1e-7},
        }

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    
def v12_binary(**kwargs):
    num_classes = getattr(kwargs,'num_classes', 10)
    return MobileNet(num_classes)