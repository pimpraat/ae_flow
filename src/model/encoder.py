import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Encoder(nn.Module):
    """Encoder block as described in the original paper. 

    "For the encoder, we chose ImageNet- pretrained Wide ResNet-50-2 (Zagoruyko & Komodakis, 2016) as 
    the feature extractor as it shows to be effective and and provide sufficient reception field. 
    
    The encoder contains four modules, and the feature map of the fourth block were chosen as the extracted features. 
    In more detail, each image is resized to 256 × 256 and the size of the extracted feature map is 16 × 16 with 1024 channels."
    (Zhao et al., 2023)

    We skip the avg_pool, fc layer and last block to get the feature map with the correct dimensions.

    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.model_p = nn.Sequential(*list(torchvision.models.wide_resnet50_2(pretrained=True).children())[:-3]) #exclude avg_pool + fc layers + last block??
    
    def forward(self, x):
        x = self.model_p(x)
        return x