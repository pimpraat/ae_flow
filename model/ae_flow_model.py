import torch.nn as nn
import torch
from nflows.distributions import normal
import torchmetrics
from torchvision.utils import make_grid, save_image
import torchvision

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

        # loss = -log_z - self.log_jac_det
        # loss = -loss.mean()/(16 * 16 * 1024)

        # Jan's proposal: should be fine since it is defined as the negative ll
        loss = log_z + self.log_jac_det
        loss = -loss.mean()/(16 * 16 * 1024)
        return loss
    
    def get_anomaly_score(self, _beta, original_x, reconstructed_x):
        Sflow = - self.get_flow_loss(return_logz=True)
        Srecon = - torchmetrics.functional.structural_similarity_index_measure(reduction=None, preds=reconstructed_x, target=original_x)

        print(f"Sflow: {Sflow}, Srecon:{Srecon}")
        return _beta * Sflow + (1-_beta)*Srecon
    
    #TODO: Finish implementation of this function
    # A function used to (easily) sample the same set of images. Using 
    def sample_images(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        grid = make_grid([self.sample_images, self.model(self.sample_images.to(device)).squeeze(dim=1)], nrow = 2)
        images = torchvision.transforms.ToPILImage()(grid)
        return images
    
    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device
