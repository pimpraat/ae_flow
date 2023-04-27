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
        model = torchvision.models.wide_resnet50_2(pretrained=True) #weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.model_p = nn.Sequential(*list(model.children())[:-3]) #exclude avg_pool + fc layers + last block?? #TODO: CHECK!
    def forward(self, x):
        """ 
        Inputs:

        Outputs:

        """
        # print(self.model_p)
        # print(f"Shape of x after encoder pass: {x.shape}")

        # i = 0
        # for layer in self.model_p:
        #     x = layer(x)
        #     print(f"Layer {i}: {x.size()}")
        #     i += 1

        # print(f"Shape of x after encoder pass: {x.shape}")
        x = self.model_p(x)
        # assert(x.shape == (64,1024,16,16))
        return x