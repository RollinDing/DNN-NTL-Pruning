#/bin/bash

# Evaluate  ADMM using a for loop when source and target are different
source_set=('mnist' 'mnistm' 'svhn' 'usps' 'syn')
target_set=('mnist' 'mnistm' 'svhn' 'usps' 'syn')
# target_set=('mnist' 'mnistm' 'svhn' 'usps' 'syn' 'cifar10' 'stl')
arch='resnet18'
rho=0.005
alpha=1
lr=1e-3
epochs=20
finetune_ratio=0.1
prune_method='admm-ntl'
seed=1
image_size=32

for source in ${source_set[@]}
do
    for target in ${target_set[@]}
    do
        # Only run if source and target are different
        if [ ${source} != ${target} ]
        then
            python src/evaluate/evaluate_ntl_performance.py data/ --arch=${arch} --source=${source} --target=${target} \
                --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} --prune-method=${prune_method} --seed=${seed} --image-size=${image_size}
        fi
    done
done