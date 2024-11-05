import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, Lambda, Normalize
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from typing import Union
from minde.scripts.helper import SynthetitcDataset
from sklearn.preprocessing import StandardScaler
from collections.abc import Iterable

from tqdm import tqdm
from minde.libs.preprocessing_utils import _array_to_tensor


class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
def pop_elem_i(encodings , i =[]  ):
    encodings = encodings.copy()
    return{
        key : encodings[key] for key in encodings.keys() if ( key in i ) == False
    } 
    
def deconcat(z,var_list,sizes):
    try:
        dim_0_sizes = []
        for size in sizes:
            if isinstance(size, Iterable):
                dim_0_sizes.append(size[0])
            else:
                dim_0_sizes.append(size)
        data = torch.split(z, dim_0_sizes, dim=1)
    except:
        raise ValueError(f"Size mismatch. Provided sizes are {sizes} but z has shape {z.shape}")
    return {var: data[i] for i, var in enumerate(var_list)}



def concat_vect(encodings):
    if len(list(encodings.values())[0].shape) == 2:
        return torch.cat(list(encodings.values()),dim = -1)
    else:
        return torch.cat(list(encodings.values()),dim = 1)
    


def unsequeeze_dict(data):
        for key in data.keys():
            if data[key].ndim == 1 :
                data[key]= data[key].view(data[key].size(0),1)
        return data


def cond_x_data(x_t,data,mod):

    x = x_t.copy()
    for k in x.keys():
        try:
            if k !=mod:
                x[k]=data[k] 
        except:
            raise ValueError(f"Key {k} not found in data. Available keys are {data.keys()}")
    return x



def marginalize_data(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k !=mod:
            if fill_zeros:
                x[k]=torch.zeros_like(x_t[k] ) 
            else:
                x[k]=torch.randn_like(x_t[k] )
    return x


def marginalize_one_var(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k ==mod:
            if fill_zeros:
                x[k]=torch.zeros_like(x_t[k] ) 
            else:
                x[k]=torch.randn_like(x_t[k] )
    return x


def minus_x_data(x_t, mod,fill_zeros=True):
        x = x_t.copy()
        for k in x.keys():
                if k ==mod:
                    if fill_zeros:
                        x[k]=torch.zeros_like(x_t[k] ) 
                    else:
                        x[k]=torch.rand_like(x_t[k] )
        return x

def _expand_mask(mask, i, size):
    # check that size is not iterable
    if not hasattr(size, '__iter__'):
        return mask[:, i].view(mask.shape[0], 1).expand(mask.shape[0], size)
    else:
        sizes = list(size)
        expanded_mask = mask[:, i]
        for i, s in enumerate(sizes):
            expanded_mask = expanded_mask.unsqueeze(-1)
        return expanded_mask.expand(mask.shape[0], *sizes)


def expand_mask(mask, var_sizes):
        return torch.cat([
            _expand_mask(mask, i, size) for i, size in enumerate(var_sizes)
        ], dim=1)
        
        
def get_samples(test_loader,device,N=10000):
    
    var_list = list(test_loader.dataset[0].keys())
    
    data = {var: torch.Tensor().to(device) for var in var_list}
    for batch in tqdm(test_loader, desc="Getting samples", leave=False):
            for var in var_list:
                data[var] = torch.cat([data[var], batch[var].to(device)])
    return {var: data[var][:N,:] for var in var_list}

def array_to_dataset(x: Union[np.array,jnp.array], y: Union[np.array,jnp.array]):
    
    x = _array_to_tensor(x)
    y = _array_to_tensor(y)
    dataset = SynthetitcDataset([x, y])
    return dataset
