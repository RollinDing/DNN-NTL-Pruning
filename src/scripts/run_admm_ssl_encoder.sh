#/bin/bash

# Run ADMM
arch='resnet50'
source='cifar10'
target='stl'
rho=0.01
alpha=1
lr=1e-3
epochs=20
finetune_ratio=0.1
batch_size=16
image_size=32

python src/prune/ssl_model_pruning.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --batch-size=${batch_size} --finetune-ratio=${finetune_ratio} --image-size=${image_size}
