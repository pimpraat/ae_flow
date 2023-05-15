import torch.nn as nn
import torch
from nflows.distributions import normal
import torchmetrics
from torchvision.utils import make_grid, save_image
import torchvision

from model.decoder import Decoder
from model.encoder import Encoder
from model.flow import FlowModule
import numpy as np

class AE_Flow_Model(nn.Module):
    def __init__(self, subnet_architecture='conv_like', custom_comptutation_graph=False, n_flowblocks=8):

        super(AE_Flow_Model, self).__init__()
        self.encoder = Encoder() #.to(memory_format=torch.channels_last)
        self.flow = FlowModule(subnet_architecture=subnet_architecture, custom_computation_graph=custom_comptutation_graph, n_flowblocks=n_flowblocks)
        self.decoder = Decoder() #.to(memory_format=torch.channels_last)

        self.sample_images_normal = []
        self.sample_images_abnormal = []

    def forward(self, x):
        z = self.encoder(x)
        self.z_prime, self.log_jac_det = self.flow(z)
        reconstructed_x = self.decoder(self.z_prime)
        return reconstructed_x
    
    def get_reconstructionloss(self, _x, recon_x):
        return nn.functional.mse_loss(recon_x, _x)
    
    def get_flow_loss(self, return_logz=False, bpd = False):
        shape = self.z_prime.shape[1:]
        log_z = normal.StandardNormal(shape=shape).log_prob(self.z_prime)
        if return_logz: return log_z / np.prod(shape).mean()
        log_p = log_z + self.log_jac_det
        nll = -log_p
        # Most people use bpp (bits per dimension) instead of nll; both UVA tutorial and every implementation I found online.
        if bpd:
            return (nll * np.log2(np.exp(1)) / np.prod(shape)).mean()
        # We don't because we're cool.
        return nll.mean()
    
    def get_anomaly_score(self, _beta, original_x, reconstructed_x):
        log_z = self.get_flow_loss(return_logz=True)
        # Sflow = - torch.exp(log_z)
        Sflow = -log_z
        Srecon = torchmetrics.functional.structural_similarity_index_measure(reduction=None, preds=reconstructed_x, target=original_x)
        # print(f"Sflow: {Sflow}, Srecon:{Srecon} in get anomaly score")
        return _beta * Sflow + (1-_beta)*Srecon
    
    def sample_images(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        grid = make_grid(self.sample_images, nrow = 1)
        images = torchvision.transforms.ToPILImage()(grid)
        return images
