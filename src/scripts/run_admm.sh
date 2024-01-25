#/bin/bash

# Run ADMM
arch='vgg11'
source='mnist'
target='usps'
rho=1e-1
alpha=1e3
lr=1e-4
epochs=10
finetune_ratio=0.1

python src/prune/admm.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs}
