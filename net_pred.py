import sys

import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from typing import Dict, Tuple, Optional, Callable

from diffusionprocess import GaussDiffusionSimple


    

def get_nllloss() -> Callable:
    """Computes the mean nll. """
    return nn.NLLLoss()    

    
def get_Y_hat_correct(Y_hat: Tensor, Y: Tensor) -> float:
    """Computes the number of correct predictions. 
    
    If `Y.shape == 1` it is assumed that `Y` stores the true class. Otherwise 
    it is assumed that `Y` is a probability vector of the same shape as `Y_hat`. 
    """
    assert len(Y_hat.shape) == 2, "Y_hat needs to have shape [bs, c]"
    
    if len(Y.shape) == 1:
        Y_hat_correct = torch.sum(torch.argmax(Y_hat, dim = 1) == Y)
    else:
        assert Y_hat.shape == Y.shape,  \
            "Y_hat and Y need to have the same shape" 
        Y_hat_correct = (Y_hat.argmax(dim = 1) == Y.argmax(dim = 1)).sum()

    return Y_hat_correct

class NetPred():
    """ Classification network class. """

    def __init__(self, model_pred: nn.Module, 
        model_diff: Optional[nn.Module], 
        diffusion: Optional[GaussDiffusionSimple], 
        optimizer: Optional[optim.Optimizer], 
        objective: Optional[nn.Module],
        device: str = "cpu") -> None:
        """
        None should only be used, if a subset of methods is relevant that don't
        need the respective attribute.
        """
        self.model_pred = model_pred.to(device)
        self.model_diff = model_diff.to(device) if model_diff is not None else None
        self.optimizer = optimizer
        self.objective = objective.to(device) if objective is not None else None
        self.diffusion = diffusion
        self.device    = device


    def __call__(self, *args) -> nn.Module:
        return self.model_pred(*args)
    

    def make_x_noisy(self, x_0: Tensor, timesteps: int, epsilon: Tensor
    ) -> Tensor:
        """Wrapper of `diffusion.sample_q` but all noise levels `t` are same."""
        n_x = x_0.shape[0]
        t   = torch.full((n_x, ), timesteps, device = self.device)
        return self.diffusion.sample_q(x_0, t, epsilon)


    def denoise_x(self, x_noisy: Tensor, timesteps: int) -> Tensor:
        """ Wrapper of `diffusion.sample` but only last tensor is taken. """
        zeta = self.diffusion.sample(self.model_diff, x_noisy, timesteps+1)[-1]
        return zeta
    

    def get_loss(self, X: Tensor, Y: Tensor, train: bool = True, 
        grad_clip: Optional[float] = 0.1) -> Tuple[float, float]:
        """ Computes loss and prediction.
         
        If `train == True` a parameter update is performed.
        """ 
        logit = self.model_pred(X)
        loss  = self.objective(logit, Y)
        Y_hat = F.softmax(logit, dim = 1)

        if train:
            self.model_pred.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_value_(self.model_pred.parameters(), grad_clip)
            self.optimizer.step()

        return loss, Y_hat
    

    def get_log_y_hat_from_zeta(self, Zeta: Tensor) -> Tensor:
        """ Compute `log_softmax` prediction of `Zeta`. """
        assert len(Zeta.shape) == 5, \
            "Zeta has to be of shape [bs, n_zeta, C, H, W]"
        bs, n_zeta, C, H, W = Zeta.shape
        Zeta      = Zeta.reshape(bs * n_zeta, C, H, W)
        logit     = self.model_pred(Zeta)
        log_Y_hat = F.log_softmax(logit, dim = -1)
        log_Y_hat = log_Y_hat.reshape(bs, n_zeta, -1)
        return log_Y_hat


    def get_loss_eiv(self, Zeta: Tensor, Y: Tensor, train: bool = True,
        grad_clip: Optional[float] = 0.1) -> Tuple[float, float]:
        """ Compute the loss of the Errors-in-Variables model.
         
        If `train == True` the parameters are updated.
        """

        assert len(Zeta.shape) == 5, \
            "Zeta has to be of shape [bs, n_zeta, C, H, W]"
        n_zeta = Zeta.size(1)
        log_Y_hat     = self.get_log_y_hat_from_zeta(Zeta)  
        log_Y_hat_avg = torch.logsumexp(log_Y_hat, dim = 1) - np.log(n_zeta)
        loss      = self.objective(log_Y_hat_avg, Y)
        if np.isnan(loss.item()):
            sys.exit("The loss is nan.")

        if train:
            self.model_pred.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_value_(self.model_pred.parameters(), grad_clip)
            self.optimizer.step()

        return loss, torch.exp(log_Y_hat_avg)


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
    
    
    def train_one_epoch_zeta(self, dl: DataLoader, 
        transform: Optional[Callable] = None) -> Tuple[float,float]:
        """ Trains model on dataset with denoised `Zeta`.

        The function assumes the input data of shape `(bs, n, C, H, W)`. 
        `n >= 1` is the number of images `Zeta` assigned to a label `Y`.
        The model is trained according to the Errors-in-Variables objective.

        Args:
            dl: `DataLoader` to retrieve the training data from.
            transform: Transformation of `Zeta`.
              This is is useful if the data in `dl` is differently transformed 
              as it is required for the prediction model `model_pred`.
        """        
        Y_hat_correct, n_element, loss_sum = 0, 0, 0
        self.model_pred.train()
        for Zeta, Y in dl:
            Zeta, Y = Zeta.to(self.device), Y.to(self.device)
            if transform is not None:
                bs, n, C, H, W = Zeta.shape
                Zeta = Zeta.reshape(bs * n, C, H, W)
                Zeta = transform(Zeta)
                Zeta = Zeta.reshape(bs, n, C, H, W)      
            loss, Y_hat    = self.get_loss_eiv(Zeta, Y, train = True)
            Y_hat_correct += get_Y_hat_correct(Y_hat, Y)
            n_element     += Y.size()[0]
            loss_sum      += loss.item()
        accuracy  = (Y_hat_correct / n_element).item()
        loss_sum /= n_element

        return accuracy, loss_sum
    

    def test_one_epoch_zeta(self, dl: DataLoader,
        transform: Optional[Callable] = None, n_theta: int = 1
    ) -> Tuple[float]:
        """ Test model on dataset with denoised `Zeta`.

        The function assumes the input data of shape `(bs, n, C, H, W)`. 
        `n >= 1` is the number of images `Zeta` assigned to a label `Y`.
        The model is trained according to the Errors-in-Variables objective.

        Args:
            dl: `DataLoader` to retrieve the training data from.
            transform: Transformation of `Zeta`.
              This is is useful if the data in `dl` is differently transformed 
              as it is required for the prediction model `model_pred`.
            n_theta: Number of predictions for the same input.
              If `n_theta == 1` inference in `eval()` mode is done. Otherwise
              inference in `train()` mode is done   
        """        
        assert n_theta > 0 and (n_theta - int(n_theta == 0)), \
            "n_theta has to be a positive integer"
        Y_hat_correct = 0
        data_size     = 0
        self.model_pred.eval()
        with torch.no_grad():
            for batch, (Zeta, Y) in enumerate(dl):
                Zeta, Y = Zeta.to(self.device), Y.to(self.device)
                if transform is not None:
                    bs, n, C, H, W = Zeta.shape
                    Zeta = Zeta.reshape(bs * n, C, H, W)
                    Zeta = transform(Zeta)
                    Zeta = Zeta.reshape(bs, n, C, H, W)
                if n_theta > 1:
                    self.model_pred.train()
                    loss, Y_hat = self.do_variational_inference_eiv(Zeta, Y, n_theta)
                else:
                    self.model_pred.eval()
                    loss, Y_hat = self.get_loss_eiv(Zeta, Y, train = False)
                Y_hat_correct += get_Y_hat_correct(Y_hat, Y)
                data_size     += len(Y_hat)
        accuracy   = (Y_hat_correct / data_size).item()
        accuracy   = np.round(accuracy, 4)

        return accuracy, loss.item()


    def train_one_epoch_noisy(self, dl: DataLoader, timesteps: int, 
        transform: Optional[Callable] = None, is_noisy: bool = True) -> float:
        """ Train model with noisification of images `X`.

        Args:
            dl: `DataLoader` to retrieve the training data from.
            timesteps: Noise level of data that is made noisy.
            transform: Transformation of `Zeta`.
              This is is useful if the data in `dl` is differently transformed 
              as it is required for the prediction model `model_pred`.
            is_noisy: Adds Gaussian noise to image `X` with diffusion model
              `model_diff`
        """
        Y_hat_correct, n_element, loss_sum = 0, 0, 0
        self.model_pred.train()
        for X, Y in dl:
            X, Y = X.to(self.device), Y.to(self.device)
            if is_noisy:
                epsilon = self.diffusion.get_noise(X)
                X       = self.make_x_noisy(X, timesteps, epsilon)
            if transform is not None:
                X = transform(X)
            loss, Y_hat    = self.get_loss(X, Y, train = True)
            Y_hat_correct += get_Y_hat_correct(Y_hat, Y) 
            n_element     += Y.size()[0]
            loss_sum      += loss.item()
        accuracy  = (Y_hat_correct / n_element).item()
        loss_sum /= n_element

        return accuracy, loss_sum


    def test_one_epoch_noisy(self, dl: DataLoader, timesteps: int, 
        transform: Optional[Callable] = None, n_theta: int = 1, 
        is_denoised: bool = True
    ) -> Tuple[float]:
        """ Test model with (noisy) images `X`.

        Args:
            dl: `DataLoader` to retrieve the training data from.
            timesteps: Expected noise level to start removing noise from data.
            transform: Transformation of `Zeta`.
              This is is useful if the data in `dl` is differently transformed 
              as it is required for the prediction model `model_pred`.
            is_denoised: Removes Gaussian noise from image `X` with diffusion 
              model `model_diff`.
        """
        assert n_theta > 0 and (n_theta - int(n_theta == 0)), \
            "n_theta has to be a positive integer"
        Y_hat_correct, data_size, loss_sum = 0, 0 ,0
        with torch.no_grad():
            for batch, (X, Y) in enumerate(dl):
                X, Y          = X.to(self.device), Y.to(self.device)
                epsilon       = self.diffusion.get_noise(X)
                X_noisy       = self.make_x_noisy(X, timesteps, epsilon)
                if is_denoised:
                    epsilon = self.diffusion.get_noise(X)
                    Zeta    = self.denoise_x(X_noisy, timesteps)
                else:
                    Zeta = X_noisy
                if transform is not None:
                    Zeta = transform(Zeta)
                if n_theta > 1:
                    self.model_pred.train()
                    loss, Y_hat = self.do_variational_inference(Zeta, Y, n_theta)
                else:
                    self.model_pred.eval()
                    loss, Y_hat = self.get_loss(Zeta, Y, train = False)
                Y_hat_correct += get_Y_hat_correct(Y_hat, Y)
                data_size     += len(Y_hat)
                loss_sum      += loss.item()
        accuracy   = (Y_hat_correct / data_size).item()
        accuracy   = np.round(accuracy, 4)

        return accuracy, loss_sum / data_size
   

    def do_variational_inference(self, X: Tensor, Y: Tensor, 
        n_theta: int) -> Tuple[Tensor, float]:
        """ Computes average prediction and loss of `n_theta` model calls.

        Useful if dropout is turned on in model. In this case the weight 
        distribution is approximated by variational inference with
        Bernoulli dropout by sampling over `n_theta` many draws of masked 
        model parameters.
        """
        self.model_pred.train()
        loss  = 0
        for _ in range(n_theta):
            loss_i, Y_hat_i = self.get_loss(X, Y, train = False)
            try:
                Y_hat +=  Y_hat_i / n_theta 
            except UnboundLocalError:
                Y_hat = Y_hat_i / n_theta
            loss  += loss_i / n_theta
        return loss, Y_hat


    def do_variational_inference_eiv(self, X: Tensor, Y: Tensor, 
        n_theta: int) -> Tuple[Tensor, float]:
        """ Computes average prediction and loss of `n_theta` model calls for EiV.

        Useful if dropout is turned on in model. In this case the weight 
        distribution is approximated by variational inference with
        Bernoulli dropout by sampling over `n_theta` many draws of masked 
        model parameters.
        """
        self.model_pred.train()
        loss = 0
        for _ in range(n_theta):
            loss_i, Y_hat_i = self.get_loss_eiv(X, Y, train = False)
            try:
                Y_hat +=  Y_hat_i / n_theta 
            except UnboundLocalError:
                Y_hat = Y_hat_i / n_theta
            loss  += loss_i / n_theta
        return loss, Y_hat
    
    def get_Y_hat(self, input: Tensor, n_theta: int, is_eiv: bool = True
    ) -> Tensor:
        """ Computes `n_theta` predictions for each input.
         
        The predictions are stacked in their first dimension. 

        Args:
            input: Image to predict label from.
            n_theta: Number of inferences. If `n_theta == 1` the model is in
              `eval()` mode else the model is in `train()` mode.
            is_eiv: If `True` the model performs Errors-in-Variable prediction
              else the model performs standard prediction
        """
        self.model_pred.eval() if n_theta == 1 else self.model_pred.train()
        Ys_hat = []
        for _ in range(n_theta):
            len_shape = len(input.shape)
            if is_eiv:
                assert len_shape == 5, "len(Zeta.shape) == 5 is to be " +\
                    f"expected but it got len(Zeta.shape) == {len_shape}"
                Y_hat = torch.exp(self.get_log_y_hat_from_zeta(input))
            else:
                assert len_shape == 4, "len(X.shape) == 4 is to be expected " +\
                    f"but it got len(X.shape) == {len_shape}"
                Y_hat = F.softmax(self.model_pred(input), dim = -1)
            Ys_hat.append(Y_hat)
        Ys_hat = torch.stack(Ys_hat, dim = 1)

        return Ys_hat


    def reset_weights_pred(self) -> None:
        """ Resets the weights of all `nn.Module` recursively.

        This operation applies to all `nn.Module` that have the attribute 
        `reset_parameters`, e.g. nn.Linear, nn.Conv2d.
        """
        @torch.no_grad()
        def reset(model: nn.Module):
            reset_parameters = getattr(model, "reset_parameters", None)
            if callable(reset_parameters):
                model.reset_parameters()

        self.model_pred.apply(fn = reset)


    def save_checkpoint(self, state, filename = 'checkpoint.pth.tar'):
        torch.save(state, filename)


    def update_model_pred(self, state_dict: Dict) -> None:
        self.model_pred.load_state_dict(state_dict)


    def update_model_diff(self, state_dict: Dict) -> None:
        self.model_diff.load_state_dict(state_dict)
