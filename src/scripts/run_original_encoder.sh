#/bin/bash

# Run Original Pruner
arch='resnet18'
source_set=('mnistm' 'svhn' 'usps' 'syn' 'mnist' 'stl' 'cifar10')
target_set=('mnistm' 'svhn' 'usps' 'syn' 'mnist' 'stl' 'cifar10')

lr=1e-3
epochs=20
finetune_ratio=0.01
prune_method='original'
seed=1

for source in ${source_set[@]}
do
    for target in ${target_set[@]}
    do
        # Only run if source and target are different
        if [ ${source} != ${target} ]
        then
            python src/prune/prune_encoder.py data/ --arch=${arch} --source=${source} --target=${target} \
                --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} \
                --prune-method=${prune_method} --seed=${seed}
        fi
    done
done