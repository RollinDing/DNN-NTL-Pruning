#/bin/bash

# Run ADMM
arch='resnet50'
source='cifar10'
target='stl'
rho=0.005
alpha=1
lr=1e-3
epochs=20
finetune_ratio=0.1
prune_method='admm-lda'
seed=3
image_size=32
batch_size=256

python src/prune/admm_encoder.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} \
    --prune-method=${prune_method} --seed=${seed} --image-size=${image_size} --batch-size=${batch_size}
