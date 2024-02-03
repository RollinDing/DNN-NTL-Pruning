#/bin/bash

# Run ADMM
arch='resnet18'
source='cifar10'

rho=0.0025
alpha=1
lr=1e-3
epochs=20

target_list=('mnist' 'usps' 'svhn' 'syn' 'mnistm' 'cifar10' 'stl')
# finetune_ratio_list=(0.01 0.02 0.05 0.1 0.2 0.5 1)
finetune_ratio_list=(0.0001 0.001 0.002 0.005 0.008)

for target in "${target_list[@]}"; do
    for finetune_ratio in "${finetune_ratio_list[@]}"; do
        python src/local/local_training.py data/ --arch=${arch} --source=${source} --target=${target} \
            --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio}
    done
done