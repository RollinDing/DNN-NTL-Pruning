#/bin/bash

arch='resnet18'
source='imagenette'
target='imagewoof'
rho=0.01
alpha=0.001
lr=1e-3
epochs=10
finetune_ratio=1
prune_method='admm-lda'
seed=2
image_size=224

python src/evaluate/evaluate_encoder_impact.py data/ --arch=${arch} --source=${source} --target=${target} --image-size=${image_size}\
    --rho=${rho} --alpha=${alpha} --lr=${lr} --finetune-ratio=${finetune_ratio} --epochs=${epochs}  --prune-method=${prune_method} --seed=${seed}
