#/bin/bash
# Run ADMM
arch='resnet18'
rho=0.0025
alpha=1
lr=1e-3
epochs=20

# source_list=('usps' 'svhn' 'syn' 'mnistm' 'cifar10' 'stl' 'mnist')
# target_list=('usps' 'svhn' 'syn' 'mnistm' 'cifar10' 'stl' 'mnist')
source_list=('imagewoof')
target_list=('imagenette')
image_size=224
finetune_ratio_list=(0.001 0.002 0.005 0.008 0.01 0.02 0.05 0.1 0.2 0.5 1)

for source in "${source_list[@]}"; do
    for target in "${target_list[@]}"; do
        if [ ${source} != ${target} ]
        then
            for finetune_ratio in "${finetune_ratio_list[@]}"; do
                python src/local/local_transfer.py data/ --arch=${arch} --source=${source} --target=${target} \
                    --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} --image-size=${image_size}
            done
        fi
    done
done