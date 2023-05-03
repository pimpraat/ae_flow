import torch.nn as nn
import torch
from nflows.distributions import normal
import torchmetrics


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
    
    def get_flow_loss(self, return_logz=False):
        shape = self.z_prime.shape[1:]
        log_z = normal.StandardNormal(shape=shape).log_prob(self.z_prime)
        if return_logz: return log_z

        loss = -log_z - self.log_jac_det
        loss = -loss.mean()/(16 * 16 * 1024)

        # Jan's proposal:
        # loss = log_z + self.log_jac_det
        # loss = -loss.mean()/(16 * 16 * 1024)
        return loss
    
    def get_anomaly_score(self, _beta, original_x, reconstructed_x):
        Sflow = - self.get_flow_loss(return_logz=True)
        Srecon = - torchmetrics.functional.structural_similarity_index_measure(reduction=None, preds=reconstructed_x, target=original_x)
        return _beta * Sflow + (1-_beta)*Srecon
