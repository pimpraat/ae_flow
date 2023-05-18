import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA


class FlowModule(nn.Module):
    """Module combining all blocks for the FlowModule
    """

    def __init__(self, subnet_architecture='conv_like', n_flowblocks=8):
        super(FlowModule, self).__init__()        
        self.inn = Ff.SequenceINN(1024, 16, 16)
        for k in range(n_flowblocks):
            if subnet_architecture == 'conv_like':
                self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.subnet_conv_3x3_1x1, permute_soft=False)
            if subnet_architecture == 'resnet_like':
                self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.Conv3x3_res_1x1, permute_soft=False)
                

    class Conv3x3_res_1x1(nn.Module):
        """ A subnet choice for the coupling layer in the flow module:
            "ResNet-type network with one 3 × 3 convolution layer with batch normalization and ReLU function, 
            and a shortcut connection with 1 × 1 convolution will be added as the output." (Zhao et al., 2023)
        
        """
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
        """A Subnet choice for the coupling layer in the flow module:
        "based on (Yu et al., 2021), for which each block contains two convolutional 
        layers with ReLU activation function, and the corresponding kernel size is 3 × 3 and 1 × 1 respectively." (Zhao et al., 2023)
        """
        return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                            nn.Conv2d(256,  c_out, 1))


    def forward(self, x):
        z, log_jac_det = self.inn(x)
        return z, log_jac_det
    

    def reverse(self,z):
        x_rev, log_jac_det_rev = self.inn(z, rev=True)
        return x_rev, log_jac_det_rev
    