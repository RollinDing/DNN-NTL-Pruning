#/bin/bash

# Run ADMM
arch='vgg11'
source='mnist'
target='mnistm'
rho=0.01
alpha=0.001
lr=1e-4

python src/prune/admm.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr}
