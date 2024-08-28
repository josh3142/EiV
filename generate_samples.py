import os
from pathlib import Path

import torch

import hydra 
from omegaconf import DictConfig 

from models.model import get_model
from datasets.dataset import get_transform
from diffusionprocess import GaussDiffusionSimple
from utils_diffusion import get_betas
from plot import plot


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:

    path = "results/" + \
        f"{cfg.data.name}/{cfg.diff.model.name}{cfg.diff.model.param.dim_mults}" + \
        f"/loss{cfg.diff.optim.loss}/" + \
        f"seed{cfg.seed}/lr{cfg.diff.optim.lr}/bs{cfg.diff.optim.batch_size}"
    path_imgs = os.path.join(path, "imgs")
    Path(path_imgs).mkdir(parents = True, exist_ok = True)
    path_checkpoint = (path + cfg.diff.checkpoint.load.path 
        + cfg.diff.checkpoint.load.name)

    # get model
    model = get_model(cfg.diff.model.name, 
        **(dict(cfg.diff.model.param) | dict(cfg.data.param))).to(cfg.device)
    checkpoint = torch.load(path_checkpoint, map_location = cfg.device)
    model.load_state_dict(checkpoint["state_dict"])
    
    betas     = get_betas(cfg.diffusion.betas, 
        timesteps = cfg.diffusion.timesteps).to(cfg.device)
    diffusion = GaussDiffusionSimple(
        cfg.diffusion.timesteps, betas)

    x_T = torch.randn((cfg.inference.n_img, cfg.data.param.n_channel, 
        cfg.data.param.dim, cfg.data.param.dim), device = cfg.device)

    samples = diffusion.sample(model, x_T)
    trafo   = get_transform(cfg.data.name, cfg.data.param.dim, reverse = True) 
    imgs = [[trafo(samples[-1][idx_row + idx_col * cfg.inference.n_row].cpu()) 
            for idx_row in range(cfg.inference.n_row)] 
        for idx_col in range(cfg.inference.n_col)]
    plot(imgs, title = f'step {checkpoint["step"]}', 
        cmap = cfg.plot.cmap, 
        save = os.path.join(path_imgs, 
            f'test_img{checkpoint["step"]}.' + cfg.plot.extension))

if __name__ == "__main__":
    run_main()

