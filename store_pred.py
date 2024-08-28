# needed for deterministic behaviour
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import DataLoader
import numpy as np

import hydra 
from omegaconf import DictConfig 

from utils import make_deterministic
from models.model import get_model as get_model_diff
from models.model import get_model_pred
from datasets.dataset import get_dataset, get_transform, get_dataset_zeta, get_transform_pred
from net_pred import NetPred
from diffusionprocess import GaussDiffusionSimple
from utils_diffusion import get_betas


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    make_deterministic(cfg.seed)
    torch.set_num_threads(16)

    path = "results_pred/" + \
        f"{cfg.data.name}/{cfg.pred.model.name}{cfg.diff.model.name}" + \
        f"/n_zeta{cfg.pred.n_zeta.train}"
    path += f"/seed{cfg.seed}"

    betas = get_betas(cfg.diffusion.betas, 
        timesteps = cfg.diffusion.timesteps).to(cfg.device)

    # initialize dataset and dataloader
    data_test = get_dataset(cfg.data.name, cfg.data.path, 
        train = False,
        transform = get_transform(cfg.data.name, 
            cfg.data.param.dim, reverse = False) )
    dl_test   = DataLoader(data_test, 
        batch_size = cfg.pred.optim.batch_size, shuffle = False, 
        num_workers = cfg.pred.optim.n_workers, 
        drop_last = False)

    data_test_zeta = get_dataset_zeta(cfg.data.name, 
        path = cfg.data.path_zeta_test,
        train = False,
        t = cfg.pred.timesteps.train, 
        n_zeta = cfg.pred.n_zeta.test, 
        n_zeta_mean = False, 
        transform = get_transform(cfg.data.name, cfg.data.param.dim,
            reverse = False) 
        )
    dl_test_zeta   = DataLoader(data_test_zeta, 
        batch_size = cfg.pred.optim.batch_size, shuffle = False, 
        num_workers = cfg.pred.optim.n_workers, drop_last = False)
 
    data_test_zeta_mean = get_dataset_zeta(cfg.data.name,
        path = cfg.data.path_zeta_test,
        train = False,
        t = cfg.pred.timesteps.train, 
        n_zeta = cfg.pred.n_zeta.test, 
        n_zeta_mean = True,
        transform = get_transform(cfg.data.name, cfg.data.param.dim,
            reverse = False) 
        )
    dl_test_zeta_mean   = DataLoader(data_test_zeta_mean, 
        batch_size = cfg.pred.optim.batch_size, shuffle = False, 
        num_workers = cfg.pred.optim.n_workers, drop_last = False)

    model_diff = get_model_diff(cfg.diff.model.name, 
        **(dict(cfg.diff.model.param) | dict(cfg.data.param)))

    model_pred_name = (str(cfg.pred.timesteps.train) + cfg.data.name + \
                    ("000" + str(cfg.pred.epoch.end - 1))[-4:] + \
                    cfg.pred.checkpoint.load.pred.name_suffix)

    # predictions Ys_hat_zeta
    # initialize predictive model class
    model_pred_zeta = get_model_pred(cfg.pred.model.name, 
        **(dict(cfg.pred.model.param) | dict(cfg.data.param) | 
           {"n_zeta": cfg.pred.n_zeta.test} | {"n_zeta_mean": False}))
    
    net_pred_zeta = NetPred(
        model_pred = model_pred_zeta,
        model_diff = model_diff,
        optimizer  = None,
        objective  = None,
        diffusion  = GaussDiffusionSimple(cfg.diffusion.timesteps, betas),
        device     = cfg.device
    )
    # load model for timestep
    checkpoint_pred = torch.load(
        os.path.join(path + "/checkpoints/", model_pred_name),
        map_location = cfg.device)
    net_pred_zeta.update_model_pred(checkpoint_pred["state_dict"])

    Y_hat_zeta, Y_zeta = [], []
    with torch.no_grad():
        for i, (Zeta, Y) in enumerate(dl_test_zeta):
            Zeta = Zeta.to(cfg.device)
            bs, n, C, H, W = Zeta.shape
            Zeta = Zeta.reshape(bs * n, C, H, W)
            Zeta = get_transform_pred(
                cfg.data.name, cfg.data.param.dim, train = False)(Zeta)
            Zeta = Zeta.reshape(bs, n, C, H, W) 
            Y_hat_zeta.append(net_pred_zeta.get_Y_hat(Zeta, 
                cfg.pred.n_theta, is_eiv = True).cpu())
            Y_zeta.append(Y)
    Y_hat_zeta = torch.cat(Y_hat_zeta, dim = 0)
    Y_zeta = torch.cat(Y_zeta, dim = 0)

    log_Y_hat_zeta     = torch.log(Y_hat_zeta)
    log_Y_hat_zeta_avg = (log_Y_hat_zeta.logsumexp(dim = (1, 2)) - \
        np.log(cfg.pred.n_theta * cfg.pred.n_zeta.test))
    print(f"Accuracy of Y_hat_zeta: " + 
        f"{((log_Y_hat_zeta_avg).argmax(-1) == Y_zeta).sum() / len(Y_zeta) * 100:.2f}%")

    # predictions Ys_hat_zeta_mean
    # initialize predictive model class
    model_pred_zeta_mean = get_model_pred(cfg.pred.model.name, 
        **(dict(cfg.pred.model.param) | dict(cfg.data.param) | 
           {"n_zeta": cfg.pred.n_zeta.test} | {"n_zeta_mean": True}))
    
    net_pred_zeta_mean = NetPred(
        model_pred = model_pred_zeta_mean,
        model_diff = model_diff,
        optimizer  = None,
        objective  = None,
        diffusion  = GaussDiffusionSimple(cfg.diffusion.timesteps, betas),
        device     = cfg.device
    )

    # load model for timestep
    path_zeta_mean = f"results_pred/{cfg.data.name}/" + \
        f"{cfg.pred.model.name}{cfg.diff.model.name}" + \
        f"/n_zeta{cfg.pred.n_zeta.train}mean/seed{cfg.seed}"
    checkpoint_pred = torch.load(
        os.path.join(path_zeta_mean + "/checkpoints/", model_pred_name),
        map_location = cfg.device)
    net_pred_zeta_mean.update_model_pred(checkpoint_pred["state_dict"])

    Y_hat_zeta_mean, Y_zeta_mean = [], []
    with torch.no_grad():
        for i, (Zeta, Y) in enumerate(dl_test_zeta_mean):
            Zeta = Zeta.to(cfg.device)
            bs, n, C, H, W = Zeta.shape
            assert n == 1, "zeta_mean should have only one averaged zeta"
            Zeta = Zeta.reshape(bs * n, C, H, W)
            Zeta = get_transform_pred(
                cfg.data.name, cfg.data.param.dim, train = False)(Zeta)
            Zeta = Zeta.reshape(bs, n, C, H, W) 
            Y_hat_zeta_mean.append(net_pred_zeta_mean.get_Y_hat(
                Zeta, cfg.pred.n_theta, is_eiv = True).cpu())
            Y_zeta_mean.append(Y)
    Y_hat_zeta_mean = torch.cat(Y_hat_zeta_mean, dim = 0)
    Y_zeta_mean = torch.cat(Y_zeta_mean, dim = 0)

    log_Y_hat_zeta_mean     = torch.log(Y_hat_zeta_mean)
    log_Y_hat_zeta_avg_mean = (log_Y_hat_zeta_mean.logsumexp(dim = (1, 2)) - \
        np.log(cfg.pred.n_theta))
    print(f"Accuracy of Y_hat_zeta_mean: " +
        f"{((log_Y_hat_zeta_avg_mean).argmax(-1) == Y_zeta_mean).sum() / len(Y_zeta_mean) * 100:.2f}%")
   
    # predictions Ys_hat_x
    # initialize predictive model class
    model_pred_x = get_model_pred(cfg.pred.model.name, 
        **(dict(cfg.pred.model.param) | dict(cfg.data.param) | 
           {"n_zeta": 0} | {"n_zeta_mean": False}))
    
    net_pred_x = NetPred(
        model_pred = model_pred_x,
        model_diff = model_diff,
        optimizer  = None,
        objective  = None,
        diffusion  = GaussDiffusionSimple(cfg.diffusion.timesteps, betas),
        device     = cfg.device
    )

    # load model for timestep
    path_x = f"results_pred/{cfg.data.name}/{cfg.pred.model.name}{cfg.diff.model.name}" + \
        f"/n_zeta0/seed{cfg.seed}"
    checkpoint_pred = torch.load(
        os.path.join(path_x + "/checkpoints/", model_pred_name),
        map_location = cfg.device)
    net_pred_x.update_model_pred(checkpoint_pred["state_dict"])

    Y_hat_x, Y_x = [], []
    with torch.no_grad():
        for i, (X, Y) in enumerate(dl_test):
            X = X.to(cfg.device)
            epsilon = net_pred_x.diffusion.get_noise(X).to(cfg.device)
            X_noisy = net_pred_x.make_x_noisy(X, cfg.pred.timesteps.train, epsilon)
            X_noisy = get_transform_pred(
                cfg.data.name, cfg.data.param.dim, train = False)(X_noisy)
            Y_hat_x.append(net_pred_x.get_Y_hat(X_noisy, 
                cfg.pred.n_theta, is_eiv = False).cpu())
            Y_x.append(Y)
    Y_hat_x = torch.cat(Y_hat_x, dim = 0)
    Y_x = torch.cat(Y_x, dim = 0)

    log_Y_hat_x     = torch.log(Y_hat_x)
    log_Y_hat_x_avg = log_Y_hat_x.logsumexp(dim = 1) - np.log(cfg.pred.n_theta)
    print(f"Accuracy of Y_hat_x: " +
        f"{((log_Y_hat_x_avg).argmax(-1) == Y_x).sum() / len(Y_x) * 100:.2f}%")

    name_npz = path + f"/{cfg.data.name}_n_theta{cfg.pred.n_theta}_n_zeta" + \
        f"{cfg.pred.n_zeta.test}_t{cfg.pred.timesteps.train}seed{cfg.seed}.npz"
    np.savez(name_npz, 
        Y_hat_zeta = Y_hat_zeta, Y_hat_zeta_mean = Y_hat_zeta_mean, 
        Y_hat_x = Y_hat_x, Y_zeta = Y_zeta, Y_zeta_mean = Y_zeta_mean, Y_x = Y_x)
    
    file = np.load(name_npz)
    Y_hat_zeta_load = file["Y_hat_zeta"]
    Y_hat_zeta_mean_load = file["Y_hat_zeta_mean"]
    Y_hat_x_load = file["Y_hat_x"]

    assert np.array_equal(Y_hat_zeta, Y_hat_zeta_load) 
    assert np.array_equal(Y_hat_zeta_mean, Y_hat_zeta_mean_load) 
    assert np.array_equal(Y_hat_x, Y_hat_x_load) 

if __name__ == "__main__":
    run_main()

