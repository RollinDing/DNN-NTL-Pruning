"""
Implementation of model non-transferable pruning algorithm on self supervised models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import logging
import time
import pickle
from copy import deepcopy
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vgg import PrunableVGG, PrunableResNet18

from prune.pruner import load_base_model
from utils.args import get_args
from utils.data import *

def load_ssl_model(args):
    # load the pretrained model trained with self-supervised learning
    method = "simclr"
    model_path = f"{args.arch}-{method}-{args.source}.tar" 
    model = torch.load(model_path)
    return model

def main():
    # load args 
    args = get_args()
    num_classes = 10
    if args.arch == 'vgg11':
        # Load the pretrained model 
        model = torchvision.models.vgg11(pretrained=True)
        # change the output layer to 10 classes (for digits dataset)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    elif args.arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)

    source_domain = args.source
    target_domain = args.target
    finetune_ratio = args.finetune_ratio