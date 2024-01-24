#/bin/bash

arch='vgg11'
source='mnist'
target='usps'
rho=0.01
alpha=0.001
lr=1e-4
finetune_ratio=0.1

python src/evaluate/evaluate_structural_impact.py data/ --arch=${arch} --source=${source} --target=${target} \
    --rho=${rho} --alpha=${alpha} --lr=${lr} --finetune-ratio=${finetune_ratio}
