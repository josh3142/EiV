import torch
from torch import Tensor, nn
import torch.nn.functional as F

from typing import Callable, List, Optional
import PIL.Image

from tqdm.auto import tqdm


class GaussDiffusionSimple():

    def __init__(self, timesteps: int, betas: Tensor) -> None:

        assert timesteps == len(betas), "You need timesteps many betas, " + \
            f"but there {len(betas)} many betas but {timesteps} many timesteps."    
        self.timesteps  = timesteps
        self.betas      = betas
        self.alphas     = 1 - self.betas
        self.alphas_bar = self.get_alphas_bar()
        self.posterior_variance = self.get_posterior_variance()


    def get_alphas_bar(self) -> Tensor:
        alphas_bar = torch.cumprod(self.alphas, axis = 0)        
        return alphas_bar


    def get_posterior_variance(self) -> Tensor:
        alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value = 1.0)
        return self.betas * (1. - alphas_bar_prev) / (1. - self.alphas_bar)


    def get_noise(self, x_0: Tensor) -> Tensor:
        """ Adds Gaussian noise of size `x_0`. """
        epsilon = torch.randn_like(x_0)
        if x_0.get_device() != -1:
            epsilon = epsilon.to(x_0.get_device())
        return epsilon


    def extract_at_t(self, X: Tensor, t: Tensor, shape) -> float:
        """ Extract t-th elements of `X`.

        These X_t are reshaped in a `Tensor` of form 
        [batch_size, 1, ..., 1] where `len([1, ..., 1]) == len(shape) - 1`
        """
        batch_size = t.shape[0]
        # gets the t-th value of X from the last axis of X 
        X_t    = X.gather(-1, t)
        X_t    = X_t.reshape(batch_size, *((1,) * (len(shape) - 1)))
        return X_t


    def sample_q(self, x_0: Tensor, t: Tensor, epsilon: Tensor) -> Tensor:
        """ Forward/ diffusion process with Gaussian noise.

        Args:
            x_0: Input (not denoised) image.
            t: Noise level to which image is denoised.
            epsilon: Gaussian noise of same shape as `x_0`.

        Returns:
            Noisy image according to diffusion process
        """
        sqrt_alphas_bar           = torch.sqrt(self.alphas_bar)
        one_minus_sqrt_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        sqrt_alphas_bar_t = self.extract_at_t(sqrt_alphas_bar, t, x_0.shape)
        sqrt_one_minus_alphas_bar_t = self.extract_at_t(
            one_minus_sqrt_alphas_bar, t, x_0.shape
        )
 
        return sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * epsilon

    def get_epsilon(self, model, x, t):
        """ The diffusion model estimates the noise to denoise the image `x`. """
        return model(x,t)

    def estimate_mu(self, model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
        """ Compute mean of Gaussian distribution of backward diffusion process.

        Args:
            model: Diffusion model to perform backward diffusion process.
            x: Noisy image.
            t: Current noise level of the image `x`.
        """
        sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)
        betas_t = self.extract_at_t(self.betas, t, x.shape)
        sqrt_one_minus_alphas_bar_t = self.extract_at_t(
            sqrt_one_minus_alphas_bar, t, x.shape)
        sqrt_recip_alphas_t = self.extract_at_t(
            torch.sqrt(1. / self.alphas), t, x.shape)
        
        epsilon = self.get_epsilon(model,x,t)  
        mu = sqrt_recip_alphas_t * (
            x - betas_t * epsilon / sqrt_one_minus_alphas_bar_t)

        return mu


    @torch.no_grad()
    def sample_p(self, model: nn.Module, x: Tensor, t: Tensor, 
        idx_t: int) -> float:
        """ Denoises image `x` from `t` to noise level `t-1`.

        Returns:
            Removes on noise level from `x` to obtain `x_prev` at noise level
            `t-1`
        """
        mu_hat      = self.estimate_mu(model, x, t)

        if idx_t == 0:
            x_prev =  mu_hat
        else:
            posterior_variance_t = self.extract_at_t(self.posterior_variance, 
                t, x.shape)
            epsilon  = self.get_noise(x)
            x_prev =  mu_hat + torch.sqrt(posterior_variance_t) * epsilon 
        return x_prev

    @torch.no_grad()
    def sample_p_t_steps(self, model: nn.Module, x_t: Tensor, 
        timesteps_start: int) -> List[Tensor]:
        """ Computes iteratively denoised image `x_0` while starting at `x_t`.
        """
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        n_samples = x_t.shape[0]
        x_ts = []
        
        for idx in tqdm(reversed(range(0, timesteps_start)), 
            desc = 'sampling loop time step', total = timesteps_start,
            disable = True):
            t = torch.full((n_samples, ), idx, device = device, 
                dtype = torch.long)
            x_t = self.sample_p(model, x_t, t, idx)
            x_ts.append(x_t)
        return x_ts


    @torch.no_grad()
    def sample(self, model: nn.Module, x_t: Tensor, 
        timesteps_start: Optional[int] = None) -> List[Tensor]:
        """ Wrapper function of `sample_p_t_steps`. """
        if timesteps_start is None:
            timesteps_start = self.timesteps
        return self.sample_p_t_steps(model, x_t, timesteps_start)
    

def get_img_from_tensor(x: Tensor, transform: Callable) -> PIL.Image:
    """ Converts a Tensor into a image. 
    
    The expected input shape is [batch_size, n_channel, n_x, n_x]. To convert 
    the Tensor into an PIL.Image the batch dimension is removed.
    """
    return transform(x.squeeze(0))


