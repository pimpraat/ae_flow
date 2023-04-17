import torch.nn as nn

from encoder import *
from decoder import *
from flow import *

class AE_Flow_Model(nn.Module):
    def __init__(self, z_dim=8):
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
            x - Batch of input images. Shape: [B,C,H,W]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
            z - Batch of latent codes. Shape: [B,z_dim]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        x = self.encoder(x)
        reconstructed_x = self.decoder(x)

        #######################
        # END OF YOUR CODE    #
        #######################
        return reconstructed_x