#/bin/bash

arch='resnet18'
source='imagenette'
target='imagewoof'
rho=0.005
alpha=1
lr=1e-3
epochs=20
finetune_ratio=1
prune_method='admm-lda'
image_size=224
batch_size=256
seeds=(1 2 3 4 5)

for seed in "${seeds[@]}"; do
    python src/evaluate/evaluate_encoder_impact.py data/ --arch=${arch} --source=${source} --target=${target} --image-size=${image_size}\
        --rho=${rho} --alpha=${alpha} --lr=${lr} --finetune-ratio=${finetune_ratio} --epochs=${epochs}  --prune-method=${prune_method} --seed=${seed}
done