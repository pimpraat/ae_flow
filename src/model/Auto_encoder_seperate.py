import torch.nn as nn
import torchmetrics

from model.decoder import Decoder
from model.encoder import Encoder

class AE_Model(nn.Module):
    """Main module for the AE with Normalized Flow module

    Args:
        subnet_architecture: which subnet to use for the coupling layer in the (flow) architecture
        n_flowblocks: Number of flow blocks in the architecture
    """
    
    def __init__(self):

        super(AE_Model, self).__init__()
        self.encoder = Encoder() 
        self.decoder = Decoder()
        self.sample_images_normal = []
        self.sample_images_abnormal = []


    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x
    

    def get_reconstructionloss(self, _x, recon_x):
        """Returnm the MSE loss for two inputs"""
        return nn.functional.mse_loss(recon_x, _x)
    
    
    def get_anomaly_score(self, original_x, reconstructed_x):
        """

        Args:
            original_x (Tensor): the original input image with shape [B, 3, 256, 256]
            reconstructed_x (Tensor): the reconstructed image with shape [B, 3, 256, 256]

        Returns:
            anomaly score: the anomaly score as proposed in the paper
        """        
        Srecon = torchmetrics.functional.structural_similarity_index_measure(reduction=None, preds=reconstructed_x, target=original_x)
        return Srecon