#/bin/bash

arch='resnet50'
source='cifar10'
target='stl'
rho=0.1
alpha=0.001
lr=1e-3
epochs=10
finetune_ratio=1.0
model_name_list=('simclr')
# model_name_list=('moco-v2' 'deepcluster-v2' 'byol' 'infomin' 'moco-v1' 'swav' 'insdis' 'pcl-v1' 'pcl-v2')
seeds=(1 2 3 4 5)
batch_size=32
image_size=32
sparsity=0.99
prune_method='admm-lda'


for seed in "${seeds[@]}"; do
    for model_name in "${model_name_list[@]}"; do
        python src/evaluate/evaluate_ssl_encoder_impact.py data/ --arch=${arch} --source=${source} --target=${target} --prune-method=${prune_method} --model-name=${model_name}\
            --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --batch-size=${batch_size} --finetune-ratio=${finetune_ratio} --image-size=${image_size}\
            --seed=${seed} --sparsity=${sparsity}
    done
done