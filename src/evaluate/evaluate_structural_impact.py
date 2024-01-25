# Evaluate the impact of the pruned sparse structure to the model transferability
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import logging
import time
from copy import deepcopy
import numpy as np
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vgg import PrunableVGG

from prune.pruner import load_base_model
from prune.admm import ADMMPruner

from utils.args import get_args
from utils.data import *


def load_dataset(args, domain, finetune_ratio):
    # Load the target dataset
    if domain == 'mnist':
        trainloader, testloader = get_mnist_dataloader(args, ratio=finetune_ratio)
    elif domain == 'cifar10':
        trainloader, testloader = get_cifar_dataloader(args, ratio=finetune_ratio)
    elif domain == 'usps':
        trainloader, testloader = get_usps_dataloader(args, ratio=finetune_ratio)
    elif domain == 'svhn':
        trainloader, testloader = get_svhn_dataloader(args, ratio=finetune_ratio)
    elif domain == 'mnistm':
        trainloader, testloader = get_mnistm_dataloader(args, ratio=finetune_ratio)
    elif domain == 'syn':
        trainloader, testloader = get_syn_dataloader(args, ratio=finetune_ratio)
    elif domain == 'stl':
        trainloader, testloader = get_stl_dataloader(args, ratio=finetune_ratio)
    
    return trainloader, testloader

def finetune_sparse_model(model, mask_dict, trainloader, testloader, nepochs=30, lr=0.001):
    # Only fine-tune the unfrozen parameters
    optimizer = torch.optim.SGD([param for name, param in model.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(nepochs):
        total_loss = 0.0
        count = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, mask_dict)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            count += len(labels)
            optimizer.step()
        print(f"Epoch {epoch}: {total_loss/count}")

def evaluate_sparse_model(model, mask_dict, testloader):
    # Fine-tune the model using trainloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model using testloader
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, mask_dict)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Evaluate Accuracy: {correct/total}")

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

    source_domain = args.source
    target_domain = args.target
    finetune_ratio = args.finetune_ratio

    # Load the dataset
    if source_domain == 'mnist':
        source_trainloader, source_testloader = get_mnist_dataloader(args, ratio=finetune_ratio)
    elif source_domain == 'cifar10':
        source_trainloader, source_testloader = get_cifar_dataloader(args, ratio=finetune_ratio)
    elif source_domain == 'usps':
        source_trainloader, source_testloader = get_usps_dataloader(args, ratio=finetune_ratio)
    elif source_domain == 'svhn':
        source_trainloader, source_testloader = get_svhn_dataloader(args, ratio=finetune_ratio)
    elif source_domain == 'mnistm':
        source_trainloader, source_testloader = get_mnistm_dataloader(args, ratio=finetune_ratio)
    elif source_domain == 'syn':
        source_trainloader, source_testloader = get_syn_dataloader(args, ratio=finetune_ratio)
    elif source_domain == 'stl':
        source_trainloader, source_testloader = get_stl_dataloader(args, ratio=finetune_ratio)

    # Load the target dataset
    if target_domain == 'mnist':
        target_trainloader, target_testloader = get_mnist_dataloader(args, ratio=finetune_ratio)
    elif target_domain == 'cifar10':
        target_trainloader, target_testloader = get_cifar_dataloader(args, ratio=finetune_ratio)
    elif target_domain == 'usps':
        target_trainloader, target_testloader = get_usps_dataloader(args, ratio=finetune_ratio)
    elif target_domain == 'svhn':
        target_trainloader, target_testloader = get_svhn_dataloader(args, ratio=finetune_ratio)
    elif target_domain == 'mnistm':
        target_trainloader, target_testloader = get_mnistm_dataloader(args, ratio=finetune_ratio)
    elif target_domain == 'syn':
        target_trainloader, target_testloader = get_syn_dataloader(args, ratio=finetune_ratio)
    elif target_domain == 'stl':
        target_trainloader, target_testloader = get_stl_dataloader(args, ratio=finetune_ratio)

    # Load the pretrained model from saved state dict
    model_path = f'saved_models/{source_domain}_to_{target_domain}/admm_model.pth'
    mask_path  = f'saved_models/{source_domain}_to_{target_domain}/admm_mask.pth'
    admm_pickle_path = f'saved_models/{source_domain}_to_{target_domain}/admm_pruner.pkl'

    pruned_model = torch.load(model_path)
    mask_dict = torch.load(mask_path)
    # admm_pruner = ADMMPruner(pruned_model, source_trainloader, target_trainloader, args)

    # mask_dict = torch.load(mask_path)
    # Compute the model sparsity using the mask_dict
    total_params = 0
    total_nonzero_params = 0
    for name, param in pruned_model.named_parameters():
        total_params += param.numel()
        if name in mask_dict:
            total_nonzero_params += torch.sum(mask_dict[name]).item()
    print(f"Model sparsity: {1-total_nonzero_params/total_params}")

    # Evaluate the model
    print("Evaluate the model on source domain")
    source_model = deepcopy(pruned_model)
    # finetune_sparse_model(source_model, mask_dict, source_trainloader, source_testloader, lr=1e-3)
    evaluate_sparse_model(source_model, mask_dict, source_testloader)

    print("Evaluate the model on target domain")
    model = load_base_model(model, 'vgg11', source_domain, source_trainloader, source_testloader)
    target_model = PrunableVGG(model)

    finetune_sparse_model(target_model, mask_dict, target_trainloader, target_testloader, lr=1e-3)
    evaluate_sparse_model(target_model, mask_dict, target_testloader)




if __name__ == "__main__":
    # Set the random seed for reproducible experiments
    torch.manual_seed(1)
    np.random.seed(1)
    main()
