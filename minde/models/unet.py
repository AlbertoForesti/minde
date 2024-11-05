import torch
import torch.nn as nn
from diffusers import UNet2DModel
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import copy

# UNet backbone for diffusion
class UNet(nn.Module):
    def __init__(self, sample_size=64, in_channels=3, out_channels=3, latent_dim=64, nb_var=2, norm_num_groups=8, name="mod"):
        super(UNet, self).__init__()
        self.name =name
        # UNet from diffusers library
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(latent_dim, latent_dim * 2, latent_dim * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            norm_num_groups=norm_num_groups,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(nb_var, 1),
            nn.GELU(),
            nn.Linear(1, 1)
        )


    def forward(self, x, timestep, condition=None):
        # Pass the latent through the U-Net for denoising
        timestep = self.time_mlp(timestep).squeeze(-1)
        ret = self.unet(x, timestep)["sample"]
        return ret