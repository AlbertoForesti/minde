import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import Normalize
import jax.numpy as jnp

def _array_to_tensor(x, preprocessing="rescale", dtype=torch.float32):
    if isinstance(x, jnp.ndarray):
        x = np.array(x)
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array or convertible to one.")
    
    if preprocessing == "rescale":
        if len(x.shape) == 2:
            x = StandardScaler(copy=True).fit_transform(x)
        else:
            x = torch.tensor(x, dtype=dtype)
            x = Normalize(0.5, 0.5)(x)
            if len(x.shape) == 3:
                x = x.unsqueeze(1) # Add channel dimension for greyscale images
            return x

    return torch.tensor(x, dtype=dtype)