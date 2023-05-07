import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, act_fn=nn.ReLU):
        """
        
        Inputs:
             
        """
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            act_fn(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),padding=1),
            act_fn(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            act_fn(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            act_fn(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            act_fn(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            act_fn(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(2,2), stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Inputs:
            
        Outputs:
            
        # """
        x = self.net(x)
        return x