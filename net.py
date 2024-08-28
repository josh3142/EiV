import torch
from torch import Tensor, nn, optim
import numpy as np

from typing import Callable, List, Dict, Optional
import time

from diffusionprocess import GaussDiffusionSimple


def get_time(fct):
    """ Yields executation time in addition to the result of the fct. """
    def wrap_fct(*args, **kwargs):
        t1 = time.time()
        result = fct(*args, **kwargs)
        t2 = time.time()
        delta_t = np.round(t2 - t1, 2)
        return result, delta_t
    return wrap_fct


def approx_standard_normal_cdf(x: Tensor) -> Tensor:
    """ Approximation of CDF of standard Gaussian distribution.

    Approximation based on Page:
    "Approximating to the Cumulative Normal function and its Inverse for use 
    on a Pocket Calculator."
    """
    z = torch.sqrt(2. / np.pi) * (x + 0.044715 * torch.pow(x, 3))
    return 0.5 * (1. + torch.tanh(z))


def discretized_gaussian_log_likelihood(x: Tensor, means: Tensor, 
    log_scales: Tensor) -> Tensor:
    """ Function assumes data as integers [0, 255] rescaled to [-1, 1]. """
    assert x.shape == means.shape == log_scales.shape
    epsilon = 1e-12

    x_centered = x - means 
    inv_std    = torch.exp(-log_scales)
    plus_in    = inv_std * (x_centered + 1. / 255)
    minus_in   = inv_std * (x_centered - 1. / 255)
    cdf_plus   = approx_standard_normal_cdf(plus_in)
    cdf_minus  = approx_standard_normal_cdf(minus_in)
    cdf_delta  = cdf_plus - cdf_minus
    log_cdf_plus = torch.log(torch.max(cdf_plus, epsilon))
    log_one_minus_cdf_minus = torch.log(torch.max(1. - cdf_minus, epsilon))

    log_probs = torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_minus, 
            torch.log(max(cdf_delta, epsilon)))
    )
    assert log_probs.shape == x.shape
    return log_probs 


class NetDiff():
    """ Diffusion model class. """

    def __init__(self, model: nn.Module, optimizer: Optional[optim.Optimizer], 
    objective: Optional[nn.Module], diffusion: GaussDiffusionSimple, 
    device: str = "cpu") -> None:
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.objective = objective
        self.diffusion = diffusion
        self.device    = device


    def __call__(self, *args) -> nn.Module:
        return self.model(*args)
    

    def get_loss(self, x_0: Tensor, t: Tensor) -> Callable:
        x_T     = self.diffusion.get_noise(x_0)
        x_noisy = self.diffusion.sample_q(x_0, t, x_T)
        x_T_hat = self.model(x_noisy, t)
        loss    = self.objective(x_T, x_T_hat)

        return loss


    @get_time
    def train_one_step(self, X: Tensor, grad_clip: Optional[float] = None, 
        warmup: Optional[int] = None, lr_max: Optional[float] = None, 
        step: Optional[float] = None
    ) -> float:
        """ Perform one training iteration """
        self.model.train()
        X  = X.to(self.device)
        bs = X.size(0)
        t  = torch.randint(0, self.diffusion.timesteps, (bs, ), 
            device = self.device).long()    
        loss = self.get_loss(X, t)
        
        # perform one backward pass
        self.model.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)
        if warmup > 0:
            if lr_max is None or step is None:
                raise ValueError("lr_max or step cannot be None if warmup is set")
            for param_g in self.optimizer.param_groups:
                param_g["lr"] = lr_max * np.minimum(step / warmup, 1.)
        self.optimizer.step()

        return loss.item()
    
    def test_one_step(self, X: Tensor) -> float:
        self.model.eval()
        X  = X.to(self.device)
        bs = X.size(0)
        t  = torch.randint(0, self.diffusion.timesteps, (bs, ), 
            device = self.device).long()    
        loss = self.get_loss(X, t)

        return loss.item()
    

    @torch.no_grad()
    def sample(self, x_t: Tensor) -> List[Tensor]:
        """ Wrapper function of `diffusion.sample`. """
        return self.diffusion.sample(self.model, x_t)
    

    def make_x_noisy(self, x_0: Tensor, timesteps: int, 
        epsilon: Tensor) -> Tensor:
        """Wrapper of `diffusion.sample_q` but all noise levels `t` are same."""
        n_x = x_0.shape[0]
        t   = torch.full((n_x, ), timesteps, device = self.device)
        return self.diffusion.sample_q(x_0, t, epsilon)
    

    def denoise_x(self, x_noisy: Tensor, timesteps: int) -> Tensor:
        """ Wrapper of `diffusion.sample` but only last tensor is taken. """
        zeta = self.diffusion.sample(self.model, x_noisy, timesteps + 1)[-1]
        return zeta

    
    def get_zeta_from_x(self, X: Tensor, timesteps: int, n_zeta: int) -> Tensor:
        """ Generate `n_zeta` `Zeta` from `X`.

        The function works in two steps:
            1. Adding noisy to `X` to construct a noisification `X_noisy`.
            2. Denoise `X_noisy` `n_zeta` times to obtain `n_zeta` `Zeta`.
        """                    
        bs, C, H, W = X.shape
        epsilon = self.diffusion.get_noise(X)
        X_noisy = self.make_x_noisy(X, timesteps, epsilon)
        X_noisy = X_noisy.unsqueeze(1)
        assert X_noisy.shape == (bs, 1, C, H, W)
        X_noisy = X_noisy.expand(bs, n_zeta, C, H, W)
        X_noisy = X_noisy.reshape(-1, C, H, W)
        Zeta = self.denoise_x(X_noisy, timesteps)
        
        return Zeta.reshape(bs, n_zeta, C, H, W)


    def save_checkpoint(self, state, filename = 'checkpoint.pth.tar'):
        torch.save(state, filename)


    def update(self, state_dict: Dict) -> None:
        self.model.load_state_dict(state_dict)
