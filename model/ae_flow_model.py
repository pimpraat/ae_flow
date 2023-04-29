import torch.nn as nn

from model.decoder import Decoder
from model.encoder import Encoder
from model.flow import FlowModule

class AE_Flow_Model(nn.Module):
    def __init__(self):
        """
       
        Inputs:
              
        """

        super(AE_Flow_Model, self).__init__()
        self.encoder = Encoder()
        self.flow = FlowModule()
        self.decoder = Decoder()

    def forward(self, x):
        """
        Inputs:

        Outputs:

        """
    

        z = self.encoder(x)
        # print(f"Shape of input for the decoder: {x.shape}")
        z_prime = self.flow(z)
        reconstructed_x = self.decoder(z_prime)

        return reconstructed_x
    
    def get_reconstructionloss(self, _x, recon_x):
        return nn.functional.mse_loss(recon_x, _x)
    
    def get_flow_loss():
        pass