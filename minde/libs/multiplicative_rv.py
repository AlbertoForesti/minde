from mutinfo.distributions.base import CorrelatedUniform
import scipy.stats
import numpy as np
import torch

class MultiplicativeRV:

    def __init__(self, samples, governing_rv, mutual_information) -> None:
        self.samples = samples
        self.governing_rv = governing_rv
        self.mutual_information = mutual_information

    def get_random_samples(self, n_samples):
        if isinstance(self.samples, np.ndarray):
            return self.samples[np.random.choice(len(self.samples), size=n_samples)], self.samples[np.random.choice(len(self.samples), size=n_samples)]
        elif isinstance(self.samples, torch.Tensor):
            return self.samples[torch.randint(0, len(self.samples), (n_samples,))].cpu().numpy(), self.samples[torch.randint(0, len(self.samples), (n_samples,))].cpu().numpy()
        else:
            if isinstance(self.samples, torch.utils.data.Dataset):
                self.samples = torch.utils.data.DataLoader(self.samples, batch_size=1, shuffle=False)
            elif isinstance(self.samples, torch.utils.data.DataLoader):
                self.samples = self.samples
            else:
                raise ValueError(f"Samples must be a numpy array, torch tensor, torch dataset or torch dataloader. Got {type(self.samples)}")
            return self.get_samples_from_dataloader(n_samples), self.get_samples_from_dataloader(n_samples)
    
    def get_samples_from_dataloader(self, n_samples):
        indices = torch.randperm(len(self.samples))[:n_samples]
        subset = torch.utils.data.Subset(self.samples.dataset, indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=n_samples, shuffle=False)
        samples, _ = next(iter(dataloader))
        return samples.cpu().numpy()
    
    def rvs(self, *args, **kwargs):
        x, y = self.governing_rv.rvs(*args, **kwargs)
        samples_x, samples_y = self.get_random_samples(len(x))
        for _ in range(samples_x.ndim - x.ndim):
            x = x[:, np.newaxis]
        for _ in range(samples_y.ndim - y.ndim):
            y = y[:, np.newaxis]

        return samples_x * x, samples_y * y