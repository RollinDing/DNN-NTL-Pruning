#/bin/bash

# Run ADMM using a for loop when source and target are different
source_set=('mnist' 'mnistm' 'svhn' 'usps' 'syn')
target_set=('mnistm' 'svhn' 'usps' 'syn' 'mnist')
arch='vgg11'
source='mnist'
target='usps'
rho=1e-2
alpha=0.001
lr=1e-4
epochs=10
finetune_ratio=0.1

for source in ${source_set[@]}
do
    for target in ${target_set[@]}
    do
        # Only run if source and target are different
        if [ ${source} != ${target} ]
        then
            python src/prune/admm.py data/ --arch=${arch} --source=${source} --target=${target} \
                --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio}
        fi
    done
done