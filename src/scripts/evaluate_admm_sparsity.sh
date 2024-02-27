#/bin/bash

# Evaluate  ADMM using a for loop when source and target are different
source='cifar10'
target='stl'
arch='resnet18'
rho=0.1
alpha=1
lr=1e-3
epochs=20
finetune_ratio=1
prune_method='admm-lda'
seed=3
sparsity=(0.5 0.8 0.9 0.95 0.99)

for s in "${sparsity[@]}"; do
    python src/evaluate/evaluate_encoder_impact.py data/ --arch=${arch} --source=${source} --target=${target} \
        --rho=${rho} --alpha=${alpha} --lr=${lr} --epochs=${epochs} --finetune-ratio=${finetune_ratio} \
        --prune-method=${prune_method} --seed=${seed} --sparsity=${s}
done
