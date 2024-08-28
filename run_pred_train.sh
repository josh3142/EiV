# To train EiV set n_zeta_train=n_zeta_test=5 and n_zeta_mean=False
# To train AIV set n_zeta_train=n_zeta_test=5 and n_zeta_mean=True
# To train non-EiV set n_zeta_train=n_zeta_test=0 and n_zeta_mean=False

# t_tr refers to the different noise levels T.

seed="1"
data="cifar10"
device="cuda:0"

model_param="[1,2,2,2]"

n_zeta_train=5
n_zeta_test=5
n_zeta_mean=False
n_theta=1

pred_bs=256

epoch_start=0
epoch_end=100
t_tr="120" # "1,30,60,90,120,150"
lr_pred="0.001"
model_diff=ddpm
model_pred=resnet9_dropout
p=0.5

python train_net_pred.py -m data=${data} \
    diff/model=${model_diff} pred/model=${model_pred} \
    pred.optim.lr=${lr_pred} pred.optim.batch_size=${pred_bs} \
    pred.n_zeta.train=${n_zeta_train} pred.n_zeta_mean=${n_zeta_mean} \
    pred.n_zeta.test=${n_zeta_test} pred.timesteps.train=${t_tr} \
    pred.n_theta=${n_theta} diff.model.param.dim_mults=${model_param} \
    pred.epoch.start=${epoch_start} pred.epoch.end=${epoch_end} \
    device=${device} seed=${seed} \
    pred.model.param.p=${p}