#/bin/bash

# Run ADMM using a for loop when source and target are different
# source_set=('mnistm' 'svhn' 'usps' 'syn' 'mnist' 'stl' 'cifar10')
source_set=('mnist')
target_set=('usps')
arch='resnet18'
rho=0.005
alphas=(0 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3)
regs=(0 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3)
lr=1e-3
epochs=20
finetune_ratio=0.1
prune_method='admm-lda'
seed=2


for alpha in ${alphas[@]}
do
    for reg in ${regs[@]}
    do
        for source in ${source_set[@]}
        do
            for target in ${target_set[@]}
            do
                # Only run if source and target are different
                if [ ${source} != ${target} ]
                then
                    python src/prune/admm_encoder.py data/ --arch=${arch} --source=${source} --target=${target} \
                        --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} --prune-method=${prune_method} --seed=${seed}  --reg=${reg}
                fi
            done
        done
    done
done
