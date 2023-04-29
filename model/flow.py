import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class FlowModule(nn.Module):

    def __init__(self):
            super(FlowModule, self).__init__()
            self.inn = Ff.SequenceINN(64, 1024, 16, 16)
            for k in range(8):
                self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.subnet_conv_3x3_1x1, permute_soft=True)


    def subnet_conv_3x3_1x1(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                            nn.Conv2d(256,  c_out, 1))
        

    def forward(self, x):
        z, log_jac_det = self.inn(x)
        return z, log_jac_det
    

    def reverse(self,z):
        x_rev, log_jac_det_rev = self.inn(z, rev=True)
        return x_rev, log_jac_det_rev
    