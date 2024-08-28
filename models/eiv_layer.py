import torch
from torch import nn, Tensor

from typing import Optional

def repeat_tensors(*args: Tensor, n_zeta: int = 1):
    """
    For each tensor in args repeat the slice 
    for each batch index (the first one) `n_zeta` times. 
    This is useful in combination in EIV Modelling for multiple draws.
    :param *args: Include here torch.tensor elements
    :param n_zeta: An integer >= 1
    """
    repeated_args = []
    for arg in args:
        repeated_arg = arg.repeat_interleave(n_zeta, dim = 0)
        repeated_args.append(repeated_arg)
    return repeated_args


class EIVDropout(nn.Module):
    """
    A Dropout Layer with Dropout probability `p` (default 0.5) that repeats the
    same Bernoulli mask `n_zeta` times along the batch dimension - instead of 
    taking a different one for each batch member. When evaluation `forward(x)` 
    the batch dimension of `x` (the first one) is asserted to be a multiple of 
    `n_zeta`. If `n_zeta = 1` `EIVDropout` is equivalent to `torch.nn.Dropout`.  
    :param p: A float between 0 and 1. Defaults to 0.5.  
    :param n_zeta: An integer >= 1
    """
    def __init__(self, p: int = 0.5, n_zeta: int = 1):
        super().__init__()
        self.p = p
        self.n_zeta = n_zeta
        self._train = True

    def train(self, training: bool = True):
        if training:
            self._train = True
        else:
            self._train = False

    def eval(self):
        self.train(training = False)

    def forward(self, x: Tensor):
        """
        Setting a set constructs a deterministic dropout mask. This is useful
        for testing.
        """
        if not self._train:
            return x
        else: 
            device = self.study_device(x)
            input_shape = x.shape 
            assert input_shape[0] % self.n_zeta == 0
            mask_shape = list(input_shape)
            mask_shape[0] = int(input_shape[0] / self.n_zeta)
            mask = torch.bernoulli(torch.ones(mask_shape) * (1 - self.p))\
                / (1 - self.p)
            repeated_mask = repeat_tensors(mask, n_zeta = self.n_zeta)[0]
            assert x.shape == repeated_mask.shape
            return x * repeated_mask.to(device)

    @staticmethod
    def study_device(x: Tensor):
        if x.is_cuda:
            return torch.device('cuda:' + str(x.get_device()))
        else:
            return torch.device('cpu')
