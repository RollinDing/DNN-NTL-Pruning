#/bin/bash

# Run ADMM
arch='resnet18'
source='cifar10'
target='stl'
rho=0.0025
alpha=1
lr=1e-3
epochs=20
finetune_ratio=0.1

python src/prune/admm_encoder.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio}
