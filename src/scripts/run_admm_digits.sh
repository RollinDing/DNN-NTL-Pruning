#/bin/bash

# Run ADMM using a for loop when source and target are different
source_set=('mnist' 'mnistm' 'svhn' 'usps' 'syn' 'cifar10' 'stl')
target_set=('mnistm' 'svhn' 'usps' 'syn' 'mnist' 'stl' 'cifar10')
arch='resnet18'
rho=0.005
alpha=1
lr=1e-3
epochs=20
finetune_ratio=0.1

for source in ${source_set[@]}
do
    for target in ${target_set[@]}
    do
        # Only run if source and target are different
        if [ ${source} != ${target} ]
        then
            python src/prune/admm_encoder.py data/ --arch=${arch} --source=${source} --target=${target} \
                --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio}
        fi
    done
done