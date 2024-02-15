#/bin/bash

arch='vgg11'
source='usps'
target='svhn'
rho=0.01
alpha=0.001
lr=1e-3
epochs=10
finetune_ratio=0.1
prune_method='admm-lda'
seed=2

python src/evaluate/evaluate_encoder_impact.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --finetune-ratio=${finetune_ratio} --epochs=${epochs}  --prune-method=${prune_method} --seed=${seed}
