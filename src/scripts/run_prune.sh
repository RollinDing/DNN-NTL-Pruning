#/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u src/prune/prune_model.py ../data --arch vgg13 \
	--epochs 10 -b 256 \
	--lr 1e-4 \
	--states 19 --save_dir cifar10_imp \
	--resume pretrained_models/model_best.pth.tar \