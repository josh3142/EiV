# needed for deterministic behaviour
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys

import torch
from torch.utils.data import DataLoader

import numpy as np

import hydra 
from omegaconf import DictConfig 
from pathlib import Path

from utils import make_deterministic, create_df, save_df, get_previous_data
from utils import get_loss_type
from utils_diffusion import get_betas
from models.model import get_model
from datasets.dataset import get_dataset, get_transform, InfDL
from net import NetDiff
from diffusionprocess import GaussDiffusionSimple
from ema import EMA
from plot import plot


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    # make_deterministic(cfg.seed)
    # torch.set_num_threads(16)

    print("")
    print("PRINT PARAMETER")
    print(cfg)

    path = "results/" + \
        f"{cfg.data.name}/{cfg.diff.model.name}{cfg.diff.model.param.dim_mults}" + \
        f"/loss{cfg.diff.optim.loss}/" + \
        f"seed{cfg.seed}/lr{cfg.diff.optim.lr}/bs{cfg.diff.optim.batch_size}"
    Path(path).mkdir(parents = True, exist_ok = True)
    path_checkpoint_save = path + cfg.diff.checkpoint.save.path
    Path(path_checkpoint_save).mkdir(parents = True, exist_ok = True)
    path_test_img = path + "/test"
    Path(path_test_img).mkdir(parents = True, exist_ok = True)

    df_name_start = f"trainingsresults{cfg.diff.step.start}.csv"
    df_name_end   = f"trainingsresults{cfg.diff.step.end}.csv"

    # initialize diffusion class
    betas = get_betas(cfg.diffusion.betas, 
        timesteps = cfg.diffusion.timesteps).to(cfg.device)
    diffusion = GaussDiffusionSimple(cfg.diffusion.timesteps, betas)

    # get model, initialize objective, optimizer and net class
    model = get_model(cfg.diff.model.name, 
        **(dict(cfg.diff.model.param) | dict(cfg.data.param)))
    init_lr   = cfg.diff.optim.lr * cfg.diff.optim.batch_size / 256   
    objective = get_loss_type(cfg.diff.optim.loss)
    optimizer = torch.optim.Adam(model.parameters(), init_lr,
            betas        = (cfg.diff.optim.adam.beta1, cfg.diff.optim.adam.beta2),
            eps          = cfg.diff.optim.adam.eps)             
    ema       = EMA(model.parameters(), decay = cfg.diff.optim.decay, device = cfg.device)
    net       = NetDiff(model, optimizer = optimizer, objective = objective, 
        diffusion = diffusion, device = cfg.device)
    
    # initialize dataset and dataloader
    data_train = get_dataset(cfg.data.name, cfg.data.path, train = True,
            transform = get_transform(cfg.data.name, cfg.data.param.dim))
    dl_train   = DataLoader(data_train, batch_size = cfg.diff.optim.batch_size,
        shuffle = True, num_workers = cfg.diff.optim.n_workers, drop_last = True)
    dl_iter    = InfDL(dl_train)

    if cfg.diff.step.start != 0:
        try:
            path_checkpoint = path + cfg.diff.checkpoint.load.path
            name_checkpoint = path_checkpoint + cfg.diff.checkpoint.load.name
            checkpoint = torch.load(name_checkpoint, map_location = cfg.device)
            state_dict = checkpoint["state_dict"]
        except FileNotFoundError:
            print(f"{name_checkpoint} doesn't exist")
            sys.exit()
        net.update(state_dict)
        steps, losses_train, losses_test, lrs, times = get_previous_data(
            os.path.join(path, df_name_start))
    else:
        steps, losses_train, losses_test, lrs, times = [], [], [], [], []
        fids = []

    for step in range(cfg.diff.step.start, cfg.diff.step.end):
        # train one step
        X, _ = dl_iter.get_next_item() 
        loss_train, time = net.train_one_step(
            X, 
            grad_clip = cfg.diff.optim.grad_clip,
            warmup    = cfg.diff.optim.warmup,
            lr_max    = init_lr,
            step      = step)
        ema.update(net.model.parameters())

        # test one step
        with torch.no_grad():
            ema.store(net.model.parameters())
            ema.copy_param_to(model.parameters())
            X, _       = dl_iter.get_next_item()
            loss_test  = net.test_one_step(X)
            fid        = np.nan
            ema.restore(net.model.parameters())

        if step%100 == 0:
            steps.append(step)
            losses_train.append(np.round(loss_train, 6))
            losses_test.append(np.round(loss_test, 6))
            fids.append(np.round(fid, 2))
            lrs.append(net.optimizer.param_groups[0]['lr'])
            times.append(time)
            print(f"The test loss at step {step} is {loss_test:.4f} " + 
                f"with FID {fid:.2f}.")

        if cfg.diff.checkpoint.save.save and (step == (cfg.diff.step.end - 1) or 
        step > cfg.diff.step.end // 2 and step%cfg.diff.step.frequ_to_save == 0):
            net.save_checkpoint({
                'step': step,
                'model': cfg.diff.model.name,
                'state_dict': net.model.state_dict(),
                'state_dict_ema': ema.state_dict(), 
                'optimizer' : net.optimizer.state_dict(),
                'loss_train': loss_train,
                'fid_test': fid
            }, filename = path_checkpoint_save + f"/model{step:07d}.pth.tar") 

        try:
            if loss_test < loss_best:
                with open(path_checkpoint_save + "/best_model.txt", "w") as f:
                    f.write(
                        f"Best model at step {step} with loss {loss_test}.")
                net.save_checkpoint({
                    'step': step,
                    'model': cfg.diff.model.name,
                    'state_dict': net.model.state_dict(),
                    'state_dict_ema': ema.state_dict(), 
                    'optimizer' : net.optimizer.state_dict(),
                    'loss_train': loss_train,
                    "fid_test": fid
                }, filename = path_checkpoint_save + f"/best_model.pth.tar")
                loss_best = loss_test 
        except:
            loss_best = 10

        df = create_df(steps, losses_train, losses_test, fids, lrs, times)
        save_df(df, os.path.join(path, df_name_end), is_file_overwritten = True)


#################### for testing #####################
        if step%1000 == 0:
            net.model.eval()
            ema.store(net.model.parameters())
            ema.copy_param_to(model.parameters())

            x_T = torch.randn((cfg.inference.n_img, cfg.data.param.n_channel, 
                cfg.data.param.dim, cfg.data.param.dim), device = cfg.device)
            samples = net.sample(x_T)
            trafo   = get_transform(cfg.data.name, cfg.data.param.dim, 
                reverse = True) 
            imgs = [[trafo(samples[-1][idx_row + idx_col * cfg.inference.n_row].cpu()) 
                    for idx_row in range(cfg.inference.n_row)] 
                for idx_col in range(cfg.inference.n_col)]
            plot(imgs, title = f"step {step}", 
                cmap = cfg.plot.cmap, 
                save = os.path.join(path_test_img, 
                    f"test_img{step}." + cfg.plot.extension))

            ema.restore(net.model.parameters())

if __name__ == "__main__":
    run_main()

