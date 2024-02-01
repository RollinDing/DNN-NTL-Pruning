#/bin/bash

arch='resnet18'
source='cifar10'
target='stl'
rho=0.01
alpha=0.001
lr=1e-3
epochs=10
finetune_ratio=1.0

python src/evaluate/evaluate_ssl_encoder_impact.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --finetune-ratio=${finetune_ratio} --epochs=${epochs}
