# needed for deterministic behaviour
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import DataLoader

from torch import nn
import numpy as np

import hydra 
from omegaconf import DictConfig 
from pathlib import Path

from utils import make_deterministic, create_df_pred_train, save_df
from models.model import get_model as get_model_diff
from models.model import get_model_pred
from datasets.dataset import get_dataset, get_transform_pred, get_transform
from datasets.dataset import get_dataset_zeta
from net_pred import NetPred, get_nllloss
from diffusionprocess import GaussDiffusionSimple
from utils_diffusion import get_betas

from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, PolynomialLR


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
    Path(path + cfg.pred.checkpoint.save.path).mkdir(parents = True, exist_ok = True)
    path_test_img = path + "/test"
    Path(path_test_img).mkdir(parents = True, exist_ok = True)

    df_name = f"trainingsresults{cfg.pred.epoch.end}.csv"

    betas = get_betas(cfg.diffusion.betas, 
        timesteps = cfg.diffusion.timesteps).to(cfg.device)

    # initialize predictive model class
    model_pred = get_model_pred(cfg.pred.model.name, 
        **(dict(cfg.pred.model.param) | dict(cfg.data.param) | 
           {"n_zeta": cfg.pred.n_zeta.train} | {"n_zeta_mean": cfg.pred.n_zeta_mean}))
    
    model_diff = get_model_diff(cfg.diff.model.name, 
        **(dict(cfg.diff.model.param) | dict(cfg.data.param)))

    # initialize objective, optimizer and net class,
    init_lr   = cfg.pred.optim.lr * cfg.pred.optim.batch_size / 256   
    objective = nn.CrossEntropyLoss() if cfg.pred.n_zeta.train == 0 \
        else get_nllloss()
    optimizer = torch.optim.Adam(model_pred.parameters(), init_lr,
            betas        = (cfg.pred.optim.adam.beta1, cfg.pred.optim.adam.beta2),
            eps          = cfg.pred.optim.adam.eps,
            weight_decay = cfg.pred.optim.wd)
     
    net_pred = NetPred(
        model_pred = model_pred,
        model_diff = model_diff,
        optimizer  = optimizer,
        objective  = objective,
        diffusion  = GaussDiffusionSimple(cfg.diffusion.timesteps, betas),
        device     = cfg.device
    )
    scheduler_warmup = LinearLR(net_pred.optimizer, start_factor = 0.001,
        total_iters = (cfg.pred.epoch.end * 0.25))
    scheduler_decay  = PolynomialLR(net_pred.optimizer, 
        total_iters = int(cfg.pred.epoch.end * 1.2))
    scheduler = ChainedScheduler([scheduler_warmup, scheduler_decay])

    # initialize dataset and dataloader
    if cfg.pred.n_zeta.train == 0:
        data_train = get_dataset(cfg.data.name, cfg.data.path, 
            train = True,
            transform = get_transform(cfg.data.name, cfg.data.param.dim,
                reverse = False) 
            )
        data_test = get_dataset(cfg.data.name, cfg.data.path, 
            train = False,
            transform = get_transform(cfg.data.name, cfg.data.param.dim, 
                reverse = False) 
            )
    else:
        data_train = get_dataset_zeta(cfg.data.name, 
            path = cfg.data.path_zeta_train,
            train = True,
            t = cfg.pred.timesteps.train, 
            n_zeta = cfg.pred.n_zeta.train, 
            n_zeta_mean = cfg.pred.n_zeta_mean,                                      
            transform = get_transform(cfg.data.name, cfg.data.param.dim,
                reverse = False) 
            )
        data_test = get_dataset_zeta(cfg.data.name, 
            path = cfg.data.path_zeta_test,
            train = False,
            t = cfg.pred.timesteps.train, 
            n_zeta = cfg.pred.n_zeta.test, 
            n_zeta_mean = cfg.pred.n_zeta_mean,
            transform = get_transform(cfg.data.name, cfg.data.param.dim,
                reverse = False) 
            )
    
    dl_train   = DataLoader(data_train, batch_size = cfg.pred.optim.batch_size,
        shuffle = True, num_workers = cfg.pred.optim.n_workers, drop_last = False)
    dl_test   = DataLoader(data_test, batch_size = cfg.pred.optim.batch_size,
        shuffle = False, num_workers = cfg.pred.optim.n_workers, drop_last = False)

    accs_train, accs_test, losses_train, losses_test = [], [], [], []
    timestep = cfg.pred.timesteps.train
    for epoch in range(cfg.pred.epoch.start, cfg.pred.epoch.end): 
        if cfg.pred.n_zeta.train == 0:
            acc_train, loss_train = net_pred.train_one_epoch_noisy(
                dl_train, 
                timestep, 
                transform = get_transform_pred(
                    cfg.data.name, cfg.data.param.dim, train = True), 
                is_noisy = True)
            acc_test, loss_test = net_pred.test_one_epoch_noisy(
                dl_test, 
                timestep, 
                transform = get_transform_pred(
                    cfg.data.name, cfg.data.param.dim, train = False), 
                n_theta = cfg.pred.n_theta,
                is_denoised = False)
        else:
            acc_train, loss_train = net_pred.train_one_epoch_zeta(
                dl_train, 
                transform = get_transform_pred(
                    cfg.data.name, cfg.data.param.dim, train = True))
            acc_test, loss_test  = net_pred.test_one_epoch_zeta(
                dl_test, 
                transform = get_transform_pred(
                    cfg.data.name, cfg.data.param.dim, train = False),
                n_theta = cfg.pred.n_theta)
        scheduler.step()
            
        losses_train.append(np.round(loss_train, 6))
        accs_train.append(np.round(acc_train, 6))
        accs_test.append(np.round(acc_test, 6))
        losses_test.append(np.round(loss_test, 6))

        print(f"{epoch}: Train/Test Accuracy: {acc_train * 100:.2f}, " +
            f"{acc_test * 100:.2f}%")  

        if cfg.pred.checkpoint.save.save and (epoch == (cfg.pred.epoch.end - 1) or 
        epoch > cfg.pred.epoch.end // 2 and epoch%cfg.pred.epoch.frequ_to_save == 0):
            net_pred.save_checkpoint({
                'epoch': epoch,
                'n_zeta': cfg.pred.n_zeta.train,
                'timestep': timestep,
                'model': cfg.pred.model.name,
                'state_dict': net_pred.model_pred.state_dict(),
                'optimizer' : net_pred.optimizer.state_dict(),
            }, 
            filename = path + \
            f"/checkpoints/{timestep}{cfg.data.name}{epoch:04d}.pth.tar") 

        df = create_df_pred_train(accs_train, accs_test, 
                                  losses_train, losses_test)
        save_df(df, os.path.join(path, str(timestep) + df_name), 
            is_file_overwritten = True)
        

if __name__ == "__main__":
    run_main()