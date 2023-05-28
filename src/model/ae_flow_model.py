import torch.nn as nn
import torch
from nflows.distributions import normal
import torchmetrics
from torchvision.utils import make_grid
import torchvision

from model.decoder import Decoder
from model.encoder import Encoder
from model.flow import FlowModule
import numpy as np

class AE_Flow_Model(nn.Module):
    """Main module for the AE with Normalized Flow module

    Args:
        subnet_architecture: which subnet to use for the coupling layer in the (flow) architecture
        n_flowblocks: Number of flow blocks in the architecture
    """
    def __init__(self, subnet_architecture, n_flowblocks):

        super(AE_Flow_Model, self).__init__()
        self.encoder = Encoder() 
        self.flow = FlowModule(subnet_architecture=subnet_architecture, n_flowblocks=n_flowblocks)
        self.decoder = Decoder()
        self.sample_images_normal = []
        self.sample_images_abnormal = []

    def forward(self, x):
        z = self.encoder(x)
        self.z_prime, self.log_jac_det = self.flow(z)
        reconstructed_x = self.decoder(self.z_prime)
        return reconstructed_x
    
    def get_reconstructionloss(self, _x, recon_x):
        """Returnm the MSE loss for two inputs"""
        return nn.functional.mse_loss(recon_x, _x)
    
    def get_flow_loss(self, return_logz=False, bpd = False):
        """Calculating the flow loss as the log_z + log_jac_det

        Args:
            return_logz (bool, optional): Wheter to only return the log_z value. Defaults to False.
            bpd (bool, optional): Wheter to return the negative log probability likelihood scaled by bpd or to return it's mea . Defaults to False.

        Returns:
            nll: logarithm probability likelihood
        """
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
        """

        Args:
            _beta (float): beta parameter to define the weight
            original_x (Tensor): the original input image with shape [B, 3, 256, 256]
            reconstructed_x (Tensor): the reconstructed image with shape [B, 3, 256, 256]

        Returns:
            anomaly score: the anomaly score as proposed in the paper
        """        
        log_z = self.get_flow_loss(return_logz=True)
        Sflow = -log_z
        Srecon = -torchmetrics.functional.structural_similarity_index_measure(reduction=None, preds=reconstructed_x, target=original_x)
        return _beta * Sflow + (1-_beta)*Srecon
    
    def sample_images(self):
        """ Function to sample the (preset) images in the model during and after training"""
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        grid = make_grid(self.sample_images, nrow = 1)
        images = torchvision.transforms.ToPILImage()(grid)
        return images