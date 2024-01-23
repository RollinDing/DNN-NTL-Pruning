#/bin/bash

# Run ADMM using a for loop when source and target are different
arch='vgg11'
source_set=('cifar10' 'stl')
target_set=('stl' 'cifar10')
rho=0.01
alpha=0.001
lr=1e-4

for source in ${source_set[@]}
do
    for target in ${target_set[@]}
    do
        # Only run if source and target are different
        if [ ${source} != ${target} ]
        then
            python src/prune/admm.py data/ --arch=${arch} --source=${source} --target=${target} \
                --rho=${rho} --alpha=${alpha} --lr=${lr}
        fi
    done
done