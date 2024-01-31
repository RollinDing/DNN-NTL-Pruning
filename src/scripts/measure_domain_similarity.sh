#/bin/bash

# Run ADMM
arch='resnet18'
source='mnist'
target='stl'
rho=0.1
alpha=1e4
lr=1e-4
epochs=20
finetune_ratio=1

python src/utils/similarity.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio}
