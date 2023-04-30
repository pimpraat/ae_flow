import torch.nn as nn
import torchmetrics

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

    def get_anomaly_score(_beta, orignal_x, reconstructed_x):
        
        Sflow = 0 # negative probability density of the normalized feature
        Srecon = - torchmetrics.functional.structural_similarity_index_measure(preds=reconstructed_x, target=orignal_x)
 
        return _beta * Sflow + (1-_beta)*Srecon 
        pass

    git config --global user.email "pimpraat@gmail.com"
    git config --global user.name "Pim Praat"
