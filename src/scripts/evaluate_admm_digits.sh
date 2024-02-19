#/bin/bash

# Evaluate  ADMM using a for loop when source and target are different
source_set=('mnist')
target_set=('usps')
# target_set=('mnist' 'mnistm' 'svhn' 'usps' 'syn' 'cifar10' 'stl')
arch='vgg11'
rho=0.005
alpha=1
lr=1e-3
epochs=20
finetune_ratio=1
prune_method='admm-lda'
seed=2

for source in ${source_set[@]}
do
    for target in ${target_set[@]}
    do
        # Only run if source and target are different
        if [ ${source} != ${target} ]
        then
            python src/evaluate/evaluate_encoder_impact.py data/ --arch=${arch} --source=${source} --target=${target} \
                --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} --prune-method=${prune_method} --seed=${seed}
        fi
    done
done