import os 

import torch
from torch.utils.data import DataLoader
import numpy as np

import hydra
from omegaconf import DictConfig

from utils_diffusion import get_betas
from diffusionprocess import GaussDiffusionSimple
from net import NetDiff
from models.model import get_model 
from datasets.dataset import get_dataset, get_transform

    
@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    
    path = os.path.join("data", "train" if cfg.pred.dataset.train == True else "test")

    # load model
    model = get_model(cfg.diff.model.name, 
        **(dict(cfg.diff.model.param) | dict(cfg.data.param)))
    weight_name = os.path.join("saved_models/diff", 
        f"{cfg.data.name}_model.pth.tar")
    checkpoint = torch.load(weight_name, map_location = cfg.device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # initialize diffusion and net class
    betas = get_betas(cfg.diffusion.betas, 
        timesteps = cfg.diffusion.timesteps).to(cfg.device)
    diffusion = GaussDiffusionSimple(cfg.diffusion.timesteps, betas)
    net       = NetDiff(model, optimizer = None, objective = None, 
        diffusion = diffusion, device = cfg.device)
    
    # initialize dataset and dataloader
    data = get_dataset(cfg.data.name, cfg.data.path, 
            train = cfg.pred.dataset.train,
            transform = get_transform(cfg.data.name, cfg.data.param.dim))
    dl   = DataLoader(data, batch_size = cfg.diff.optim.batch_size,
        shuffle = False, num_workers = cfg.diff.optim.n_workers, 
        drop_last = False)

    Zeta_all, Y_all = [], []
    for X, Y in dl:
        Y    = Y.cpu().numpy()
        X    = X.to(cfg.device)
        Zeta = net.get_zeta_from_x(X, cfg.pred.timesteps.train, cfg.pred.n_zeta.test)
        Zeta = get_transform(cfg.data.name, cfg.data.param.dim, reverse = True,
            is_pil = False)(Zeta.cpu())
        Zeta_all.append(Zeta)
        Y_all.append(Y)
    Zeta_all = np.concatenate(Zeta_all, axis = 0) 
    Y_all    = np.concatenate(Y_all, axis = 0)
    assert Zeta_all.shape == (len(data),) + (cfg.pred.n_zeta.test,) + Zeta.shape[-3:]
    assert Y_all.shape == (len(data),) + Y.shape[1:]

    # store data in numpy array
    stored_data_name = os.path.join(path, 
        f"{cfg.data.name}t{cfg.pred.timesteps.train}n_zeta{cfg.pred.n_zeta.test}.npz")
    np.savez(stored_data_name, data = Zeta_all, Y = Y_all)

    # load data from file
    file = np.load(stored_data_name)
    Zeta_load, Y_load = file["data"], file["Y"]
    assert np.array_equal(Zeta_load, Zeta_all)
    assert np.array_equal(Y_load, Y_all)

if __name__ == "__main__":
    run_main()