#/bin/bash

arch='vgg11'
source='mnist'
target='usps'
rho=0.005
alpha=1
lr=1e-3
epochs=20
finetune_ratio=1
prune_method='admm-lda'
image_size=32
batch_size=256
sparsity=0.99
seed=2

python src/evaluate/evaluate_encoder_impact.py data/ --arch=${arch} --source=${source} --target=${target} --image-size=${image_size} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --finetune-ratio=${finetune_ratio} --epochs=${epochs}  --prune-method=${prune_method} --seed=${seed} --sparsity=${sparsity}
