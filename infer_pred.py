# needed for deterministic behaviour
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import hydra 
from omegaconf import DictConfig 
from pathlib import Path

from utils import make_deterministic, create_df_pred, save_df
from models.model import get_model as get_model_diff
from models.model import get_model_pred
from datasets.dataset import get_dataset, get_transform, get_transform_pred, get_dataset_zeta
from net_pred import NetPred, get_nllloss
from diffusionprocess import GaussDiffusionSimple
from utils_diffusion import get_betas


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    make_deterministic(cfg.seed)
    torch.set_num_threads(16)
    if cfg.pred.n_zeta.train == 0:
        assert cfg.pred.n_zeta.test == 0, \
        "if n_zeta_train == 0, n_zeta_test should be 0" 
    else:
        assert cfg.pred.n_zeta.test > 0, \
        "if n_zeta_train > 0, n_zeta_test should be greater than 0" 

    path = "results_pred/" + \
        f"{cfg.data.name}/{cfg.pred.model.name}{cfg.diff.model.name}" + \
        f"/n_zeta{cfg.pred.n_zeta.train}"
    if cfg.pred.n_zeta_mean:
        path += "mean"
    path += f"/seed{cfg.seed}"
    Path(path).mkdir(parents = True, exist_ok = True)

    betas = get_betas(cfg.diffusion.betas, 
        timesteps = cfg.diffusion.timesteps).to(cfg.device)

    # initialize predictive model class
    model_pred = get_model_pred(cfg.pred.model.name, 
        **(dict(cfg.pred.model.param) | dict(cfg.data.param) | 
           {"n_zeta": cfg.pred.n_zeta.test} | {"n_zeta_mean": cfg.pred.n_zeta_mean}))
    
    model_diff = get_model_diff(cfg.diff.model.name, 
        **(dict(cfg.diff.model.param) | dict(cfg.data.param)))

    net_pred = NetPred(
        model_pred = model_pred,
        model_diff = model_diff,
        optimizer  = None,
        objective  = nn.CrossEntropyLoss() if cfg.pred.n_zeta.test == 0 else \
            get_nllloss(),
        diffusion  = GaussDiffusionSimple(cfg.diffusion.timesteps, betas),
        device     = cfg.device
    )

    for timestep in cfg.pred.timesteps.inference:
        # load model for timestep
        model_pred_name = (str(timestep) + cfg.data.name + \
                        ("000" + str(cfg.pred.epoch.end - 1))[-4:] + \
                        cfg.pred.checkpoint.load.pred.name_suffix)
        checkpoint_pred = torch.load(
            os.path.join(path + "/checkpoints/", model_pred_name),
            map_location = cfg.device)
        net_pred.update_model_pred(checkpoint_pred["state_dict"])

        if cfg.pred.n_zeta.train == 0:
            # initialize dataset and dataloader
            data_test = get_dataset(cfg.data.name, cfg.data.path, 
                train = False,
                transform = get_transform(cfg.data.name, 
                    cfg.data.param.dim, reverse = False) )
            dl_test   = DataLoader(data_test, 
                batch_size = cfg.pred.optim.batch_size,
                shuffle = False, 
                num_workers = cfg.pred.optim.n_workers, 
                drop_last = False)
            # test one epoch
            acc_zeta, loss_zeta = net_pred.test_one_epoch_noisy(dl_test, 
                timestep, 
                transform =  get_transform_pred(cfg.data.name, 
                    cfg.data.param.dim, train = False),
                n_theta     = cfg.pred.n_theta,
                is_denoised = False)       
        else:
            try:
                # load dataset for timestep
                data_test_zeta = get_dataset_zeta(cfg.data.name, 
                    path = cfg.data.path_zeta_test,
                    train = False,
                    t = timestep, 
                    n_zeta = cfg.pred.n_zeta.test, 
                    n_zeta_mean = cfg.pred.n_zeta_mean, 
                    transform = get_transform(cfg.data.name, cfg.data.param.dim,
                        reverse = False) 
                    )
                dl_test_zeta   = DataLoader(data_test_zeta, 
                    batch_size = cfg.pred.optim.batch_size, shuffle = False, 
                    num_workers = cfg.pred.optim.n_workers, drop_last = False)

                # get accuracy and loss for timestep
                acc_zeta, loss_zeta = net_pred.test_one_epoch_zeta(
                    dl_test_zeta, 
                    transform = get_transform_pred(
                        cfg.data.name, cfg.data.param.dim, train = False),
                    n_theta = cfg.pred.n_theta)   
            except FileNotFoundError:
                acc_zeta, loss_zeta = np.nan, np.nan

        print(f"timestep {timestep}: Zeta: {acc_zeta * 100:.2f}%")
        
        df = create_df_pred([timestep], [acc_zeta], [np.nan], 
            [loss_zeta], [np.nan])
        save_df(df, 
            os.path.join(path, f"inference{cfg.pred.n_zeta.test}n_zeta" + \
                f"{cfg.pred.n_theta}n_theta.csv"), 
            is_file_overwritten = False)
    
if __name__ == "__main__":
    run_main()

