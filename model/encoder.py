import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(nn.Module):
    """
        Inputs:     
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.block = torchvision.models.wide_resnet50_2(pretrained=True), #weights=Wide_ResNet50_2_Weights.DEFAULT)

    def forward(self, x):
        """ 
        Inputs:

        Outputs:

        """
        # print(f"block: {self.block[0]}")
        # print(f"x: {x.shape, x}")
        x = self.block[0](x)
        x = self.block[0](x)
        x = self.block[0](x)
        x = self.block[0](x)
        assert(x.shape == (16,16,1024))
        return x