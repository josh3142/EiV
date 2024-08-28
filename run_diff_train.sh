seed=41
data="cifar10"
cmap="viridis"
device="cuda:0"
step_start=0
step_end=800000
model="ddpm"
model_param="[1,2,2,2]"
bs=128

model_loaded2="/best_model.pth.tar"

python train_net.py -m data=${data} diff/model=${model} \
    diff.model.param.dim_mults=${model_param} plot.cmap=${cmap} \
    diff.step.start=${step_start} diff.step.end=${step_end} \
    diff.optim.batch_size=${bs} device=${device} seed=${seed} 

python generate_samples.py -m data=${data} diff/model=${model} \
    diff.model.param.dim_mults=${model_param} plot.cmap=${cmap} \
    diff.checkpoint.load.name=${model_loaded2} diff.optim.batch_size=${bs} \
    device=${device} seed=${seed}

python generate_animation.py -m data=${data} diff/model=${model} \
    diff.model.param.dim_mults=${model_param} plot.cmap=${cmap} \
    diff.checkpoint.load.name=${model_loaded2} diff.optim.batch_size=${bs} \
    device=${device} seed=${seed}

