import torch.nn as nn
import torch
from nflows.distributions import normal


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
        shape = self.z_prime.shape[1:]
        log_z = normal.StandardNormal(shape=shape).log_prob(self.z_prime)
        loss = log_z + self.log_jac_det
        loss = -loss.mean()/(16 * 16 * 1024)
        return loss