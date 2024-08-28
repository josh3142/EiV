import torch
from torch import Tensor

def cosine_beta_schedule(timesteps: int, beta_0: float, beta_T: float, 
    s: float = 0.008) -> Tensor:
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_0, beta_T)


def linear_beta_schedule(timesteps: int, beta_0: float, 
    beta_T: float) -> Tensor:
    
    return torch.linspace(beta_0, beta_T, timesteps)


def quadratic_beta_schedule(timesteps: int, beta_0: float, 
    beta_T: float) -> Tensor:
    
    return torch.linspace(beta_0**0.5, beta_T**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int, beta_0: float, 
    beta_T: float) -> Tensor:

    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_T - beta_0) + beta_0


def get_betas(schedule: str, timesteps: int, beta_0: float = 0.0001, 
    beta_T: float = 0.02) -> Tensor:
    """ Generates noise schedule to perform diffusion process with. """
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps, beta_0, beta_T)
    elif schedule == "linear":
        return linear_beta_schedule(timesteps, beta_0, beta_T)
    elif schedule == "quadratic":
        return quadratic_beta_schedule(timesteps, beta_0, beta_T)
    elif schedule == "sigmoid":
        return sigmoid_beta_schedule(timesteps, beta_0, beta_T)
    else:
        raise NotImplementedError()
