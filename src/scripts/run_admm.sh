#/bin/bash

# Run ADMM
arch='resnet18'
source='mnist'
target='mnistm'
rho=0.1
alpha=1e4
lr=1e-4
epochs=20
finetune_ratio=0.1

python src/prune/admm.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs}
