import torch.nn as nn

from encoder import *
from decoder import *
from flow import *

class AE_Flow_Model(nn.Module):
    def __init__(self):
        """
       
        Inputs:
              
        """
        super(AE_Flow_Model, self).__init__()
        self.encoder = Encoder()
        self.flow = None
        self.decoder = Decoder()

    def forward(self, x):
        """
        Inputs:

        Outputs:

        """
   

        z = self.encoder(x)
        reconstructed_x = self.decoder(z)

        return reconstructed_x
    
    def get_reconstructionloss(x, recon_x):
        return torch.nn.functional.mse_loss(input=recon_x, target=x)