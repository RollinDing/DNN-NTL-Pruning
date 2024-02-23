#/bin/bash

# Run ADMM
arch='resnet50'
source='cifar10'
target='stl'
rho=1e0
alpha=1
lr=1e-4
epochs=20
finetune_ratio=0.1
prune_method='admm-lda'
model_name_list=('moco-v1', 'moco-v2', 'deepcluster-v2', 'byol', 'infomin', 'insdis', 'pcl-v1', pcl-v2', 'swav')
seed=2
batch_size=256
image_size=32

for model_name in "${model_name_list[@]}"; do
    python src/prune/ssl_model_pruning.py data/ --arch=${arch} --source=${source} --target=${target} --prune-method=${prune_method} --model-name=${model_name}\
        --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --batch-size=${batch_size} --finetune-ratio=${finetune_ratio} --image-size=${image_size} --seed=${seed}
done