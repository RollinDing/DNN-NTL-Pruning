#/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u src/prune/prune_model.py ../data --arch vgg13 \
	--epochs 10 -b 256 --pretrain True\
	--lr 1e-2 \
	--states 19 --save_dir cifar10_imp 