# To obtain posterior predictives for all three statistical models set
# n_zeta_train5, n_zeta_test=50

# t refers to the noise level T


seed="1"
data="cifar10"
device="cuda:0"

model_param="[1,2,2,2]"

n_zeta_train=5
n_zeta_test=50
n_theta=50

pred_bs=256

epoch_start=0
epoch_end=100
t=120 # "1,30,60,90,120,150" 
model_diff=ddpm
model_pred=resnet9_dropout
p=0.5

python store_pred.py -m data=${data} \
    diff/model=${model_diff} pred/model=${model_pred} \
    pred.n_theta=${n_theta} pred.optim.batch_size=${pred_bs}\
    pred.n_zeta.train=${n_zeta_train} pred.n_zeta.test=${n_zeta_test} \
    pred.timesteps.train=${t} \
    diff.model.param.dim_mults=${model_param} \
    pred.epoch.start=${epoch_start} pred.epoch.end=${epoch_end} \
    device=${device} seed=${seed} \
    pred.model.param.p=${p}
