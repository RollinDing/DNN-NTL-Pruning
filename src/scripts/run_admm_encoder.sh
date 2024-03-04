#/bin/bash

# Run ADMM
arch='resnet18'
source='cifar10'
target='stl'
rho=0.005
alpha=0.1
lr=1e-3
epochs=1
finetune_ratios=(0.01 0.05 0.1 0.2 0.4 0.5 0.8)
prune_method='admm-lda'
image_size=32
batch_size=256
seeds=(1 2 3 4 5)
sparsity=0.95

for finetune_ratio in "${finetune_ratios[@]}"; do
    for seed in "${seeds[@]}"; do
        python src/prune/admm_encoder.py data/ --arch=${arch} --source=${source} --target=${target} \
            --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} \
            --prune-method=${prune_method} --seed=${seed} --image-size=${image_size} --batch-size=${batch_size} --sparsity=${sparsity}
    done
done
