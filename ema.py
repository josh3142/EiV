import torch 
from typing import Generator, Optional, Dict

# Partially based on: https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
class EMA:
    """ Maintains the exponential moving average (EMA) of the model parameters. 
    """

    def __init__(self, params: Generator, decay: float, 
        is_n_update: bool = True, device = "cpu") -> None:
        """
        Args: 
            params: Iterable of model.parameters()
            decay: Exponential decay rate
            is_n_update: Should number of updates be used to get EMA
        """
        if decay < 0. or decay > 1.:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.n_update = 0 if is_n_update else None
        self.ema_params = [p.detach().clone().to(device) 
            for p in params if p.requires_grad]
        self.temp_params = []

    @torch.no_grad()
    def update(self, params: Generator) -> None:
        """ Update currently maintained parameters `params`.

        Call this every time the parameters are updated.

        Args:
            params: Same as `params` with which the class is initialized.
        """
        decay = self.decay
        if self.n_update is not None:
            self.n_update += 1
            decay = min(decay, (1 + self.n_update) / (10 + self.n_update))
        
        params = [p for p in params if p.requires_grad]
        for ema_p, p in zip(self.ema_params, params):
            ema_p.sub_((1. - decay) * (ema_p - p))

    def copy_param_to(self, params: Generator) -> None:
        """ Replace parameter in parameters with EMA parameters
        
        Args:
            params: Parameters to be replaced by the EMA parameters
        """
        params = [p for p in params if p.requires_grad]
        for ema_p, p in zip(self.ema_params, params):
            if p.requires_grad:
                p.data.copy_(ema_p.data)

    def store(self, params: Generator) -> None:
        """ Temporarily save params. 
        
        Temporally stored params can be retrieved by `restore`. 
        Can be used to store training parameters of the model while
        the model uses for inference (validating) the EMA `params`.

        Args:
            params: Parameters to be temporarily stored
        """
        self.temp_params = [p.clone() for p in params]

    def restore(self, params: Generator) -> None:
        """ Restore the parameters stored with the `store` method.

        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        
        Args:
            params: Parameters to be updated with the stored parameters.
        """
        for t_p, p in zip(self.temp_params, params):
            p.data.copy_(t_p.data)

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "n_update": self.n_update,
                "ema_params": self.ema_params}

    def load_state_dict(self, state_dict: Dict) -> None:
        """ Mimics pytorch's `load_state_dict`. """
        self.decay = state_dict['decay']
        self.n_update = state_dict['n_update']
        self.ema_params = state_dict['ema_params']