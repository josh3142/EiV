import os

import torch
from torch import Tensor

from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage
from torchvision.transforms import RandomHorizontalFlip, Normalize, Resize 
from torchvision.transforms import RandomGrayscale, RandomApply, RandomResizedCrop
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from typing import Optional, Callable, Tuple

import numpy as np


class DatasetZetaFromNumpy(Dataset):
    """ Generates a dataset from numpy arrays for EiV.
    
    The numpy array contains several `X` and has shape [bs, n_zeta, H, W, C].
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, 
        transform: Optional[Callable] = None):
        """
        Args:
            X: Input data.
            Y: Label of input data.
            transform: Transformation of input data.
        """
        self.X = X
        self.Y = torch.LongTensor(Y)
        self.transform = transform
        _, self.n_zeta, _, _, _ = X.shape
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tensor]:
        x, y = self.X[idx, ...], self.Y[idx, ...]
        if self.transform is not None:
            x_new = []
            for i in range(self.n_zeta):
                x_i = self.transform(x[i])
                x_new.append(x_i)
            x = torch.stack(x_new, dim = 0)

        return (x, y)
        
    def __len__(self) -> int:
        return self.Y.size(0)


class DatasetZetaFromNumpyMean(Dataset):
    """ Generates a dataset from numpy arrays for AIV.

    The numpy array contains several zetas and has shape [bs, n_zeta, H, W, C]
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, 
        transform: Optional[Callable] = None):
        """
        Args:
            X: Input data.
            Y: Label of input data.
            transform: Transformation of input data.
        """

        self.X = self._preprocess_X(X)
        self.Y = torch.LongTensor(Y)
        self.transform = transform
        _, self.n_zeta, _, _, _ = X.shape

    def _preprocess_X(self, X: np.ndarray) -> np.ndarray:
        """ Computing the mean of all `n_zeta` instances of the image. 

        Take the mean of all n_zeta and rescale the array to the range [0, 1].
        The rescaling has to be done in addition because the array is not of 
        type `np.unit8` after taking the mean.
        """
        bs, n_zeta, H, W, C = X.shape 
        X_processed = (X.mean(axis = 1, keepdims = True) / 255.).astype(np.float32)
        assert X_processed.shape == (bs, 1, H, W, C), \
            "Processed X is of wrong shape"
        assert X_processed.min() >= -1e-6, "Processed X isn't rescaled properly" 
        assert X_processed.max() <= 1 + 1e-6, "Processed X isn't rescaled properly"

        return X_processed
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tensor]:
        x, y = self.X[idx, ...], self.Y[idx, ...]
        if self.transform is not None:
            x = self.transform(x[0]).unsqueeze(0)

        return (x, y)
        
    def __len__(self) -> int:
        return self.Y.size(0)


class DatasetNumpy(Dataset):
    """ Generates a dataset from numpy arrays with integer as label."""
    def __init__(self, X: np.ndarray, Y: np.ndarray, 
        transform: Optional[Callable] = None) -> None:
        self.X         = X
        self.Y         = torch.LongTensor(Y)
        self.transform = transform
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tensor]:
        x, y = self.X[idx, ...], self.Y[idx, ...]
        if self.transform is not None:
            x = self.transform(x)

        return (x, y)
        
    def __len__(self) -> int:
        return self.Y.size(0)



def get_dataset(name: str, path: str, train: bool, transform: 
        Optional[Callable] = None) -> Dataset:
    """ Retrieve the dataset for non-EiV. """
    if name == "cifar10":
        data = CIFAR10(path, train = train, transform = transform, 
            download = True)
    elif name == "mnist":
        data = MNIST(path, train = train, transform = transform, 
            download = True)
    elif name == "cifar100":
        data = CIFAR100(path, train = train, transform = transform, 
            download = True)
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")
        
    return data

def get_dataset_zeta(name: str, path: str = "data", train: bool = True, 
    t: int = 60, n_zeta: int = 5, n_zeta_mean: bool = False,
    transform: Optional[Callable] = None) -> Dataset:
    """ Retrieve the dataset for EiV and AIV. """
    
    file_name = os.path.join(path, f"{name}t{t}n_zeta{n_zeta}.npz")
    npzfile = np.load(file_name)
    X, Y = npzfile["data"], npzfile["Y"]
    if name in ["cifar10", "cifar100", "mnist"]:
        if not n_zeta_mean:
            data = DatasetZetaFromNumpy(X, Y, transform)
        else:
            data = DatasetZetaFromNumpyMean(X, Y, transform)
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")
    
    return data


class InfDL:
    """ Generate infinite dataloader from finite `DataLoader`.

    `DataLoader` is reinitialed with `__iter__` after it reaches the last 
    element. 
    """
    def __init__(self, dl):
        self.dl = dl
        self.dl_iter = iter(self.dl)
    
    def get_next_item(self) -> Tensor:
        try: 
            x, y = next(self.dl_iter)
        except StopIteration:
            self.dl_iter = iter(self.dl)
            x, y = next(self.dl_iter)
        return x, y


def get_transform(name: str, dim: int, reverse: bool = False, 
    is_pil: bool = True) -> Callable:
    """ Data transformation for diffusion process. 
    
    Args:
        reverse: If `reverse == False` transformation for training phase is 
          applied (`Pil.Image` -> `Tensor`). Else the backward transformatio is 
          applied (`Tensor` -> `Pil.Image`).
    """
    if name == "mnist":
        transform = get_trafo_mnist(dim, reverse, is_pil)
    elif name in ["cifar10"]:
        transform = get_trafo_cifar10(dim, reverse, is_pil)
    elif name == "cifar100":
        transform = get_trafo_cifar10(dim, reverse, is_pil)
    else:
        raise NotImplementedError("A transformation for this dataset is " + \
            "implemented")

    return transform


def get_transform_pred(name: str, dim: int, train: bool = True) -> Callable:
    """ Transform diffusion network input to prediction network input.

    Transforms `Tensor` from the range [-1, 1] to [0, 1] and performs data 
    augmentation in addition.

    Note:
      This transformation is intended to be used after the transformation used
      for the diffusion process
    """
    if name == "mnist":
        transform = get_trafo_pred_mnist(dim, train)
    elif name in ["cifar10"]:
        transform = get_trafo_pred_cifar10(dim, train)
    elif name == "cifar100":
        transform = get_trafo_pred_cifar100(dim, train)
    else:
        raise NotImplementedError("A transformation for this dataset is " + \
            "implemented")

    return transform


def get_trafo_mnist(dim: int = 28, reverse: bool = True, 
    is_pil: bool = True) -> Callable:
    if not reverse:    
        transform = [
            ToTensor(), # turn into tensor of shape CHW, divide by 255
            Resize(dim),
            Lambda(lambda t: (t * 2) - 1) # adjust range to [-1, 1]
        ]
    else:
        transform = [
            Lambda(lambda t: (t - t.min()) / (t.max() - t.min())),
            # Lambda(lambda t: (t + 1) / 2), # adjust range to [0, 1]
            Lambda(lambda t: t.moveaxis(-3, -1)), #CHW to HWC
            Lambda(lambda t: t * 255.), # adjust range to [0, 255]
            Lambda(lambda t: t.numpy().astype(np.uint8))
        ]
        if is_pil:
            transform += [ToPILImage()]
    
    return Compose(transform)

def get_trafo_pred_mnist(dim = 28, train: bool = True) -> Callable:
    mu, std = (0.13066048920154572), (0.30810782313346863)
    trafo = [
        Lambda(lambda t: (t + 1) / 2), # adjust range to [0, 1]
        Normalize(mean = mu, std = std),
        Resize(dim, interpolation= InterpolationMode.BICUBIC)
     ]
    if train: 
        trafo += [
            RandomHorizontalFlip(p = 0.5),
        ]

    return Compose(trafo)
  
  
def get_trafo_cifar10(dim: int = 32, reverse: bool = True, 
    is_pil: bool = True) -> Callable:
    if not reverse:    
        transform = [
            ToTensor(), # turn into tensor of shape CHW, divide by 255
            Resize(dim),
            RandomHorizontalFlip(),
            Lambda(lambda t: (t * 2) - 1) # adjust range to [-1, 1]
        ]
    else:
        transform = [
            Lambda(lambda t: (t - t.min()) / (t.max() - t.min())),
            Lambda(lambda t: t.moveaxis(-3, -1)), #CHW to HWC
            Lambda(lambda t: t * 255.), # adjust range to [0, 255]
            Lambda(lambda t: t.numpy().astype(np.uint8))
        ]
        if is_pil:
            transform += [ToPILImage()]
    
    return Compose(transform)


def get_trafo_pred_cifar10(dim = 32, train: bool = True) -> Callable:
    mu  = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    trafo = [
        Lambda(lambda t: (t + 1) / 2), # adjust range to [0, 1]
        Normalize(mu, std)
        ]
    if train:
        trafo += [
            RandomHorizontalFlip(p = 0.5),
            RandomGrayscale(p = 0.3),
            RandomApply(
                torch.nn.ModuleList([RandomResizedCrop(dim, scale = (0.2, 1.0))]), 
                p = 0.5)
        ]

    return Compose(trafo)


def get_trafo_pred_cifar100(dim = 32, train: bool = True) -> Callable:
    mu  = (0.50707516, 0.48654887, 0.44091784)
    std = (0.26733429, 0.25643846, 0.27615047)
    trafo = [
        Lambda(lambda t: (t + 1) / 2), # adjust range to [0, 1]
        Normalize(mu, std)
        ]
    if train:
        trafo += [
            RandomHorizontalFlip(p = 0.5),
            RandomGrayscale(p = 0.3),
            RandomApply(
                torch.nn.ModuleList([RandomResizedCrop(dim, scale = (0.2, 1.0))]), 
                p = 0.5),
        ]

    return Compose(trafo)

