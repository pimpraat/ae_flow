import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(nn.module):
    """
        Inputs:     
    """
    def init(self, n_blocks=4):
        super(Encoder, self).__init__()
        block = torchvision.models.wide_resnet50_2(pretrained=True)
        self.net = nn.Sequential(block, block, block, block)

    def forward(self, x):
        """
        Inputs:

        Outputs:

        """
        x = self.net(x)
        assert(x.shape == (16,16,1024))
        return x