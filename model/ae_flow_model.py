import torch.nn as nn
import torch

from model.decoder import Decoder
from model.encoder import Encoder
from model.flow import FlowModule

class AE_Flow_Model(nn.Module):
    def __init__(self):

        super(AE_Flow_Model, self).__init__()
        self.encoder = Encoder()
        self.flow = FlowModule()
        self.decoder = Decoder()

    def forward(self, x):
        
        z = self.encoder(x)
        self.z_prime, self.log_jac_det = self.flow(z)
        reconstructed_x = self.decoder(self.z_prime)
        return reconstructed_x
    
    def get_reconstructionloss(self, _x, recon_x):
        return nn.functional.mse_loss(recon_x, _x)
    
    def get_flow_loss(self):
        loss = 0.5*torch.sum(self.z_prime**2, 1) - self.log_jac_det
        print(loss.shape)
        return loss.mean()