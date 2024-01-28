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

def finetune_sparse_model(model, mask_dict, trainloader, testloader, nepochs=50, lr=0.001):
    # Only fine-tune the unfrozen parameters
    # optimizer = torch.optim.SGD([param for name, param in model.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if param.requires_grad], lr=lr)
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

            # apply the mask to the model
            for name, param in model.named_parameters():
                if name in mask_dict:
                    param.data = param.data * mask_dict[name]
                    # set the gradient to zero
                    param.grad = param.grad * mask_dict[name]

        print(f"Epoch {epoch}: {total_loss/count}")

        # # how many percentage parameters are adjusted 
        # changed = 0
        # total = 0
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         changed += torch.sum(param.grad != 0).item()
        #         total += param.numel()
        # print(f"Percentage of changed parameters: {changed/total}")

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

def evaluate_transferability_with_ratio(model, target_trainloader, target_testloader):
    # Evaluate the model transferability with different number of fine-tuning samples
    total_train_samples = len(target_trainloader.dataset)

    # Training sample number
    train_sample_nums = [int(total_train_samples*ratio) for ratio in np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])]

    # For each train sample num, randomly select the samples and evaluate the transferability
    for train_sample_num in train_sample_nums:
        print(f"Train sample num: {train_sample_num}")
        # Randomly select the samples
        indices = np.random.choice(total_train_samples, train_sample_num, replace=False)
        train_subset = torch.utils.data.Subset(target_trainloader.dataset, indices)
        subtrainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
        # Evaluate the transferability 
        model_copy = deepcopy(model)
        evaluate_transferability(model_copy, subtrainloader, target_testloader)

def evaluate_transferability(model, target_trainloader, target_testloader):
    # Evaluate the model transferability
    # Fine-tune the model using trainloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Fine-tune the model using trainloader
    optimizer = torch.optim.SGD([param for name, param in model.named_parameters() if param.requires_grad], lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(10):
        total_loss = 0.0
        count = 0
        for inputs, labels in target_trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            count += len(labels)
            optimizer.step()
        print(f"Epoch {epoch}: {total_loss/count}")

    # Evaluate the model using testloader
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in target_testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
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

    model = load_base_model(model, 'vgg11', source_domain, source_trainloader, source_testloader)
    
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

    # evaluate_transferability_with_ratio(model, target_trainloader, target_testloader)

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
    finetune_sparse_model(source_model, mask_dict, source_trainloader, source_testloader, lr=1e-4)
    evaluate_sparse_model(source_model, mask_dict, source_testloader)


    print("Evaluate the model on target domain")
    target_model = deepcopy(pruned_model)
    # Evaluate the model transferability with different number of fine-tuning samples
    total_train_samples = len(target_trainloader.dataset)

    # Training sample number
    train_sample_nums = [int(total_train_samples*ratio) for ratio in np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_model.to(device)
    target_model.to(device)
    
    # build an all-one mask 
    all_one_mask_dict = {}
    for name, param in target_model.named_parameters():
        if name in mask_dict:
            all_one_mask_dict[name] = torch.ones_like(mask_dict[name])

    # # Replace the weights in target model with the remain weights in source model
    # for name, param in source_model.named_parameters():
    #     if name in mask_dict:
    #         target_model.state_dict()[name].data = param.data * mask_dict[name] + target_model.state_dict()[name].data * (1 - mask_dict[name])


    # For each train sample num, randomly select the samples and evaluate the transferability
    for train_sample_num in train_sample_nums:
        print(f"Train sample num: {train_sample_num}")
        # Randomly select the samples
        indices = np.random.choice(total_train_samples, train_sample_num, replace=False)
        train_subset = torch.utils.data.Subset(target_trainloader.dataset, indices)
        subtrainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
        # Evaluate the transferability 
        model_copy = deepcopy(target_model)    

        finetune_sparse_model(model_copy, mask_dict, subtrainloader, target_testloader, lr=1e-3)
        evaluate_sparse_model(model_copy, mask_dict, target_testloader)




if __name__ == "__main__":
    # Set the random seed for reproducible experiments
    torch.manual_seed(1)
    np.random.seed(1)
    main()
