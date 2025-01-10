
import torch
import math
import itertools
import numpy as np
from .importance import *
from .util import concat_vect, expand_mask
from functools import reduce
from diffusers.utils.torch_utils import randn_tensor


class VP_SDE():
    def __init__(self,
                 beta_min=0.1,
                 beta_max=20,
                 N=1000,
                 importance_sampling=True,
                 type="c",
                 weight_s_functions=True,
                 var_sizes=[1, 1]
                 ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.var_sizes = var_sizes
        self.rand_batch = False
        self.N = N
        self.T = 1
        self.importance_sampling = importance_sampling
        self.nb_var = len(self.var_sizes)
        self.weight_s_functions = weight_s_functions
        self.device = "cuda"
        self.type = type
        self.masks = self.get_masks_training()
       

    def set_device(self, device):
        self.device = device
        self.masks = self.masks.to(device)

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, t):
        # Returns the drift and diffusion coefficient of the SDE ( f(t), g(t)) respectively.
        return -0.5*self.beta_t(t), torch.sqrt(self.beta_t(t))

    def marg_prob(self, t, x):
        
        ## Returns mean and std of the marginal distribution P_t(x_t) at time t.
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        if len(x.shape) > 2:
            mean = mean.view(-1,1)
            std = std.view(-1,1)
            for i in range(len(x.shape)-2):
                mean = mean.unsqueeze(-1)
                std = std.unsqueeze(-1)
        elif len(x.shape) == 2:
            mean = mean.view(-1,1)
            std = std.view(-1,1)
        else:
            raise NotImplementedError("Shape not supported")
        try:
            ret = mean * torch.ones_like(x, device=self.device), std * torch.ones_like(x, device=self.device)
        except:
            raise ValueError(f"Shape mismatch mean={mean.shape} std={std.shape} x={x.shape}")
        return ret

    def sample(self, x_0, t):
        ## Forward SDE
        # Sample from P(x_t | x_0) at time t. Returns A noisy version of x_0.
        mean, std = self.marg_prob(t, t)
        z = torch.randn_like(x_0, device=self.device)
        for i in range(len(x_0.shape)-2):
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)
        try:
            x_t = x_0 * mean + std * z
        except:
            raise ValueError(f"Shape mismatch x_0={x_0.shape} mean={mean.shape} std={std.shape} z={z.shape}")
        return x_t, z, mean, std

    def train_step(self, data, score_net, eps=1e-5, return_denoised=False):
        """
        Perform a single training step for the SDE model.

        Args:
            data : The input data for the training step.
            score_net : The score network used for computing the score.
            eps: A small value used for numerical stability when importance sampling is Off. Defaults to 1e-5.
        Returns:
            Tensor: The loss value computed during the training step.
        """

        x_0 = concat_vect(data)

        bs = x_0.size(0)

        if self.importance_sampling:
            t = (self.sample_importance_sampling_t(
                shape=(x_0.shape[0], 1))).to(self.device)
        else:
            t = ((self.T - eps) *
                 torch.rand((x_0.shape[0], 1)) + eps).to(self.device)
        # randomly sample an index to choose a masks
        if self.rand_batch:
            i = torch.randint(low=1, high=len(self.masks)+1, size=(bs,)) - 1
        else:
            i = (torch.randint(low=1, high=len(self.masks)+1, size=(1,)) - 1 ).expand(bs)
            
        # Select the mask randomly from the list of masks to learn the denoising score functions.

        mask = self.masks[i.long(), :]
        mask_data = expand_mask(mask, self.var_sizes)
        # Variables that are not marginal
        mask_data_marg = (mask_data < 0).float()
        # Variables that will be diffused
        mask_data_diffused = mask_data.clip(0, 1)
        x_t, Z, mean, std = self.sample(x_0=x_0, t=t)

        x_t = mask_data_diffused * x_t + (1 - mask_data_diffused) * x_0
        x_t = x_t * (1 - mask_data_marg) + torch.zeros_like(x_0, device=self.device) *mask_data_marg

        output = score_net(x_t, t=t, mask=mask, std=None)
        try:
            score = output * mask_data_diffused
        except:
            raise ValueError(f"Shape mismatch output={output.shape} mask_data_diffused={mask_data_diffused.shape}")
        Z = Z * mask_data_diffused

        x_denoised = (x_t - std * score)/mean
        #Score matching of diffused data reweithed proportionnaly to the size of the diffused data.
        
        # Flatten the data
        score = score.view(bs, -1)
        mask_data_diffused = mask_data_diffused.view(bs, -1)
        Z = Z.view(bs, -1)

        total_size = score.size(1)
        n_diff=torch.sum(mask_data_diffused, dim=1)
        try:
            weight = (((total_size - n_diff) / total_size) + 1).view(bs, 1)
        except:
            raise ValueError(f"Shape mismatch total_size={total_size} n_diff={n_diff.shape} bs={bs}")
        loss = (weight * (torch.square(score - Z))).sum(1, keepdim=False)/n_diff

        if return_denoised:
            return loss, x_denoised, x_t
        
        return loss


    
    
    def get_masks_training(self):
        """
        Returns a list of masks each corresponds to a score function needed to compute MI.
        
        
        """
        if self.type=="c":
            masks= np.array([[1,-1],[1,0]]) 
        elif self.type=="j":
            masks= np.array([[1,1],[1,0],[0,1]])  
        
        if self.weight_s_functions:
            return torch.tensor(self.weight_masks(masks), device=self.device)
        else:
            return  torch.tensor(masks, device=self.device)


    def weight_masks(self, masks):
        """ Weighting the mask list so the more complex score functions are picked more often durring the training step. 
        This is done by duplicating the mask with the list of masks.
        """
        masks_w = []
  
        #print("Weighting the scores to learn ")
        for s in masks:
                nb_var_inset = np.sum(s == 1)
                for i in range(nb_var_inset):
                    masks_w.append(s)
        np.random.shuffle(masks_w)
        return np.array(masks_w)
    
    def step_pred(self, score, x, t, generator=None):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            score ():
            x ():
            t ():
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # TODO(Patrick) better comments + non-PyTorch
        # postprocess model score
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        std = std.flatten()
        while len(std.shape) < len(score.shape):
            std = std.unsqueeze(-1)
        score = -score / std

        # compute
        dt = -1.0 / len(self.timesteps)

        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        beta_t = beta_t.flatten()
        while len(beta_t.shape) < len(x.shape):
            beta_t = beta_t.unsqueeze(-1)
        drift = -0.5 * beta_t * x

        diffusion = torch.sqrt(beta_t)
        drift = drift - diffusion**2 * score
        x_mean = x + drift * dt

        # add noise
        noise = randn_tensor(x.shape, layout=x.layout, generator=generator, device=x.device, dtype=x.dtype)
        x = x_mean + diffusion * math.sqrt(-dt) * noise

        return x, x_mean
    
    def generate_samples(self, score_net, input_shape, mask, bs=512, steps = 100):
        """
        Generate samples from the SDE model.
        """
        dt = self.T / steps
        # sizes = [2, 16, 16]
        x_0 = torch.randn(bs, *input_shape, device=self.device)
        mask = mask.unsqueeze(0).expand(bs, -1)
        mask_data = expand_mask(mask, self.var_sizes)
        # Variables that are not marginal
        mask_data_marg = (mask_data < 0).float()
        # Variables that will be diffused
        mask_data_diffused = mask_data.clip(0, 1)
        # x[:,1,:,:] = 0
        try:
            x_0 = mask_data_diffused * x_0 + (1 - mask_data_diffused) * x_0
        except:
            raise ValueError(f"Shapes mismatch mask_data_diffused={mask_data_diffused.shape} x_0={x_0.shape}")
        x_t = x_0 * (1 - mask_data_marg) + torch.zeros_like(x_0, device=self.device) *mask_data_marg
        self.timesteps = torch.linspace(self.T, 0, steps + 1, device=self.device)
        for i in range(steps):
            t = self.T - i * dt
            t = torch.ones(bs, 1, device=self.device) * t
            score = score_net(x_t, t=t, mask=mask, std=None)
            x_t, x_mean = self.step_pred(score, x_t, t)
            # x[:,1,:,:] = 0
            x_t = mask_data_diffused * x_t + (1 - mask_data_diffused) * x_0
            x_t = x_t * (1 - mask_data_marg) + torch.zeros_like(x_0, device=self.device) *mask_data_marg
        # x_mean[:,1,:,:] = 0
        x_mean = mask_data_diffused * x_mean + (1 - mask_data_diffused) * x_0
        x_mean = x_mean * (1 - mask_data_marg) + torch.zeros_like(x_0, device=self.device) *mask_data_marg
        return x_mean
    
    def sample_importance_sampling_t(self, shape):
        """
        Non-uniform sampling of t to importance_sampling. See [1,2] for more details.
        [1] https://arxiv.org/abs/2106.02808
        [2] https://github.com/CW-Huang/sdeflow-light
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, T=self.T)
