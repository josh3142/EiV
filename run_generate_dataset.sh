# t refers to the different noise levels T.

seed=42
data="cifar10"
device="cuda:0"
train=True # False
t=120
n_zeta=5
model="ddpm"
model_param="[1,2,2,2]"
bs=128

python generate_dataset.py -m data=${data} diff/model=${model} \
    diff.model.param.dim_mults=${model_param} diff.optim.batch_size=${bs} \
    pred.dataset.train=${train} pred.timesteps.train=${t} \
    pred.n_zeta.test=${n_zeta} \
    device=${device} seed=${seed}

