import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.model_p = nn.Sequential(*list(torchvision.models.wide_resnet50_2(pretrained=True).children())[:-3]) #exclude avg_pool + fc layers + last block??
    
    def forward(self, x):
        x = self.model_p(x)
        return x