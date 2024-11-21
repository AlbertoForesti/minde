import scipy
import numpy as np

class MixedBernoulli:

    def __init__(self, p, rv_1, rv_2) -> None:
        self.bernoulli = scipy.stats.bernoulli(p)
        self.rv_1 = rv_1
        self.rv_2 = rv_2
    
    def rvs(self, *args, **kwargs):
        z = self.bernoulli.rvs(*args, **kwargs).reshape(-1,1)
        x1, y1 = self.rv_1.rvs(*args, **kwargs)
        x2, y2 = self.rv_2.rvs(*args, **kwargs)
        w = x1*z + x2*(1-z), y1*z + y2*(1-z)
        """raise UserWarning(f"Some examples of the random variable are: {w[:5]}\n\
                          z: {z[:5]}\n\
                          x1: {x1[:5]}\n\
                          y1: {y1[:5]}\n\
                          x2: {x2[:5]}\n\
                          y2: {y2[:5]}\n\
                          args: {args}\n\
                          kwargs: {kwargs}")"""
        return w

class ShiftedNormal:

    def __init__(self, p, mean, std, shift) -> None:
        self.normal = scipy.stats.norm(loc=mean, scale=std)
        self.bernoulli = scipy.stats.bernoulli(p)
        self.shift = shift
    
    def rvs(self, *args, **kwargs):
        x2 = self.normal.rvs(*args, **kwargs).reshape(-1,1)
        x1 = x2 + self.shift # If std is very low x1 will be very similar to shift_vector
        z = self.bernoulli.rvs(*args, **kwargs).reshape(-1,1)
        # High p -> z = 1
        shift_vector = self.shift*np.ones_like(x1)
        """raise UserWarning(f"Some examples of the random variable are:\n\
                          z: {z[:5]}\n\
                          shift_vector * z: {(shift_vector * z)[:5]}\n\
                          x1 * z + x2 * (1-z): {(x1 * z + x2 * (1-z))[:5]}\n\
                          args: {args}\n\
                          kwargs: {kwargs}")"""
        return x1 * z + x2 * (1-z), shift_vector * z