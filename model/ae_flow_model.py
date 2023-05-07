import torch.nn as nn
import torch
from nflows.distributions import normal
import torchmetrics
import numpy as np

from model.decoder import Decoder
from model.encoder import Encoder
from model.flow import FlowModule

class AE_Flow_Model(nn.Module):
    def __init__(self):

        super(AE_Flow_Model, self).__init__()
        self.encoder = Encoder()
        self.flow = FlowModule()
        self.decoder = Decoder()

        self.sample_images = [] ## Should be 3 images?

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
        # Jan's proposal: should be fine since it is defined as the negative ll
        log_p = log_z + self.log_jac_det
        nll = -log_p
        # Most people use bpp (bits per dimension) instead of nll; both UVA tutorial and every implementation I found online.
        bpd = nll * np.log2(np.exp(1)) / np.prod(shape)
        return nll.mean()
    
    def get_anomaly_score(self, _beta, original_x, reconstructed_x):
        log_z = self.get_flow_loss(return_logz=True)
        Sflow = - torch.exp(log_z)
        Srecon = - torchmetrics.functional.structural_similarity_index_measure(reduction=None, preds=reconstructed_x, target=original_x)
        return _beta * Sflow + (1-_beta)*Srecon
    
    # A function used to (easily) sample the same set of images. Using 
    def sample_images(self):
        pass

        # now save the images to file

