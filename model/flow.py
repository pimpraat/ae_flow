import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA

from FrEIA.modules import InvertibleModule
from typing import Sequence, Union
from copy import deepcopy

class FlowModule(nn.Module):

    def __init__(self, subnet_architecture='conv_like', custom_computation_graph=False, n_flowblocks=8):
        super(FlowModule, self).__init__()        

        for k in range(n_flowblocks):
            if subnet_architecture == 'conv_like':
                self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.subnet_conv_3x3_1x1, permute_soft=False)
            if subnet_architecture == 'resnet_like':
                self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.Conv3x3_res_1x1, permute_soft=False)
                

    class Conv3x3_res_1x1(nn.Module):
        def __init__(self, size_in, size_out):
            super().__init__()
            self.conv = nn.Conv2d(size_in, size_out, kernel_size=3, padding='same', bias=1)
            self.bn = nn.BatchNorm2d(size_out)
            self.relu = nn.ReLU(inplace=True)
            self.res = nn.Conv2d(size_in, size_out, kernel_size=1, bias=0)
        def forward(self, x):
            output = self.conv(x)
            output = self.bn(output)
            output = self.relu(output)
            res = self.res(x)
            return output + res



    def subnet_conv_3x3_1x1(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                            nn.Conv2d(256,  c_out, 1))


    def forward(self, x):
        z, log_jac_det = self.inn(x)
        return z, log_jac_det
    

    def reverse(self,z):
        x_rev, log_jac_det_rev = self.inn(z, rev=True)
        return x_rev, log_jac_det_rev
    
