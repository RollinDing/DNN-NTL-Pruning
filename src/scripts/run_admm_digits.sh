#/bin/bash

# Run ADMM using a for loop when source and target are different
arch='vgg11'
source_set=('mnist' 'mnistm' 'svhn' 'usps' 'syn)
target_set=('mnistm' 'svhn' 'usps' 'syn' 'mnist')
rho=0.01
alpha=0.001
lr=1e-4

for i in {0..4}
do
    source=${source_set[i]}
    target=${target_set[i]}
    if $source != $target
    then
        python src/prune/admm.py data/ --arch=${arch} --source=${source} --target=${target} \
            --rho=${rho} --alpha=${alpha} --lr=${lr}
    fi
done
