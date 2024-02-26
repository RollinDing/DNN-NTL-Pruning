#/bin/bash
# Run ADMM
arch='vgg11'
rho=0.0025
alpha=1
lr=1e-3
epochs=20
batch_size=32

source_list=('cifar10')
target_list=('stl')
# source_list=('imagewoof')
# target_list=('imagenette')
# image_size=224
image_size=32
finetune_ratio_list=(0.001 0.002 0.005 0.008 0.01 0.02 0.05 0.1 0.2 0.5 1)
seed_list=(0 1 2 3 4)

for seed in $seed_list; do
    for source in "${source_list[@]}"; do
        for target in "${target_list[@]}"; do
            for finetune_ratio in "${finetune_ratio_list[@]}"; do
                python src/local/local_training.py data/ --arch=${arch} --source=${source} --target=${target} --batch-size=${batch_size}\
                    --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} --image-size=${image_size} --seed=${seed}
            done
        done
    done
done