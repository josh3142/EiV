import numpy as np
import random
import torch
import torch.nn.functional as F

import pandas as pd

from typing import List, Tuple, Callable
import time


def make_deterministic(seed) -> None:
    random.seed(seed)   	    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only = True)


def get_loss_type(name: str = "l2") -> Callable:

    if name == 'l1':
        loss = F.l1_loss
    elif name == 'l2':
        loss = F.mse_loss
    elif name == "huber":
        loss = F.smooth_l1_loss
    else:
        raise NotImplementedError()

    return loss


def create_df_pred(timesteps: List[int], 
    acc_zetas: List[float], acc_x_ts: List[float],
    loss_zetas: List[float], loss_x_ts: List[float]) -> pd.DataFrame:
    
    df = pd.DataFrame(data =
            {"timestep": timesteps,
            "acc_zeta": acc_zetas,
            "acc_x_t": acc_x_ts,
            "loss_zeta": loss_zetas,
            "loss_x_t": loss_x_ts
            }
        )
    
    return df


def create_df_pred_train(acc_train: List[float], acc_test: List[float], 
    loss_train: List[float], losses_test: List[float]) -> pd.DataFrame:
    
    df = pd.DataFrame(data =
            {"epoch": [int(i) for i in range(len(loss_train))],
            "acc_train": acc_train, 
            "acc_test": acc_test,
            "loss_train": loss_train,
            "loss_test": losses_test
            }
        )
    
    return df

def create_df(steps: List[float], loss_train: List[float], loss_test: List[float], 
    fids: List[float], lrs: List[float], time : List[float]) -> pd.DataFrame:
    
    df = pd.DataFrame(data =
            {"step": steps,
            "loss_train": loss_train,
            "loss_test": loss_test,
            "FID": fids,
            "lr": lrs,
            "time": time
            }
        )
    
    return df
    

def save_df(df: pd.DataFrame, filename: str, 
    is_file_overwritten: bool = False) -> None:
    
    if not is_file_overwritten:
        try:
            df_loaded = pd.read_csv(filename)
            df = pd.concat([df_loaded, df], axis = 0)
        except FileNotFoundError:
            print("File does not exists.")
            print("A new file is created")
    df.to_csv(filename, index = False)


def get_previous_data(name: str) -> Tuple[List]:
    df           = pd.read_csv(name, index_col = False)
    steps        = df["steps"].tolist()
    losses_train = df["loss_train"].tolist()
    lrs          = df["lr"].tolist()
    times        = df["time"].tolist()
    
    return steps, losses_train, lrs, times

def get_timing(fct):
    def wrapper(*args, **kwargs):
        t1  = time.time()        
        res = fct(*args, **kwargs)
        t2  = time.time()
        print(f"{fct.__name__} needs {t2 - t1:.2f}s to execute.")        
        return res
    return wrapper 
