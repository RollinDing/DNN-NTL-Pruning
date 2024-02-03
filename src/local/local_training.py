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
from models.encoders import ResNetEncoder, ResNetClassifier

from prune.pruner import load_base_model
from utils.args import get_args
from utils.data import *
from itertools import cycle

def local_train(model, source_trainloader, source_testloader):
    # finetune the model on source dataset 
    print('Finetune the model on source dataset')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nepochs = 100
    model.to(device)    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # adjust the learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0
    patience = 10
    for epoch in range(nepochs):
        model.train()
        for inputs, labels in (source_trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
        scheduler.step()

        # validate the model
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for input, labels in source_testloader:
                input = input.to(device)
                labels = labels.to(device)
                outputs = model(input)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch+1}/{nepochs}, Accuracy on source dataset: {correct / total}')

        # Early stopping
        if correct / total > best_acc:
            best_acc = correct / total
            # save the best model
            print("Get a better model")
            patience=10
        else:
            patience-=1
            if patience == 0:
                print('Early stopping at epoch:', epoch+1)
                break

    print("Finish fine-tune the model, the best accuracy is:", best_acc)
    return best_acc

def main():
    # load args 
    args = get_args()
    num_classes = 10
    if args.arch == 'vgg11':
        # Load the pretrained model 
        model = torchvision.models.vgg11(pretrained=False)
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
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)

    target_domain = args.target
    finetune_ratio = args.finetune_ratio

    # Create the logger 
    log_dir = os.path.join(os.path.dirname(__file__), '../..', f'logs/{args.arch}/local')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, f'{args.target}.log')
    # The log file should clear every time
    logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO)
    # Log the time 
    logging.info(time.asctime(time.localtime(time.time())))
    # Log the args
    logging.info(args)
    # Log the source and target domain
    logging.info(f'Local training on {target_domain} dataset')

    # Load the source dataset
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

    # Train the model from scratch
    best_acc = local_train(model, target_trainloader, target_testloader)
    data_volume = len(target_trainloader.dataset)
    logging.info(f'Data ratio {finetune_ratio}, Data volume {data_volume}, Local training on {target_domain} dataset, the best accuracy is {best_acc}')

if __name__ == "__main__":
    main()