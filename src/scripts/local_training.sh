#/bin/bash

# Run ADMM
arch='resnet18'
source='cifar10'

rho=0.0025
alpha=1
lr=1e-3
epochs=20

# source_list=('usps' 'svhn' 'syn' 'mnistm' 'cifar10' 'stl' 'mnist')
target_list=('stl' )
# finetune_ratio_list=(0.001 0.002 0.005 0.008 0.01 )
finetune_ratio_list=(0.001 0.002 0.005 0.008 0.01 0.02 0.05 0.1 0.2 0.5 1)
image_size=32
batch_size=32
seed=0
# finetune_ratio_list=(0.0001 0.001 0.002 0.005 0.008)

for target in "${target_list[@]}"; do
    for finetune_ratio in "${finetune_ratio_list[@]}"; do
        python src/local/local_training.py data/ --arch=${arch} --source=${source} --target=${target} --batch-size=${batch_size}\
            --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} --image-size=${image_size} --seed=${seed}
    done
done