import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import logging
import time
from copy import deepcopy
import numpy as np
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.args import get_args
from utils.data import *

def finetune_model(model, trainloader, testloader, nepochs=30, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # fine-tune the encoder and classifier
    # optimizer = torch.optim.SGD([param for name, param in encoder.named_parameters() if param.requires_grad] + [param for name, param in classifier.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if param.requires_grad], lr=lr, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(nepochs):
        total_loss = 0.0
        count = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            features = model.features(inputs)
            features = features.view(features.size(0), -1)
            outputs = model.classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            count += len(labels)
            optimizer.step()

        print(f"Epoch {epoch}: {total_loss/count}")


def evaluate_model(model, testloader):
    # Fine-tune the model using trainloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate the model using testloader
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            features = model.features(inputs)
            features = features.view(features.size(0), -1)
            outputs = model.classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Evaluate Accuracy: {correct/total}")
    return correct/total

def main():
    # load args 
    args = get_args()
    source_domain= args.source
    target_domain= args.target
    # Load the state dict of the model
    state_dict = torch.load(f'saved_models/vgg11/NTL/{source_domain}_to_{target_domain}.pth')
    # Rename the unexpected keys in state_dict
    # Remove the 1 after classifier key
    for i in [0, 3, 6]:
        state_dict[f'classifier.{i}.weight'] = state_dict[f'classifier1.{i}.weight']
        state_dict[f'classifier.{i}.bias'] = state_dict[f'classifier1.{i}.bias']
        state_dict.pop(f'classifier1.{i}.weight')
        state_dict.pop(f'classifier1.{i}.bias')

    model = torchvision.models.vgg11()
    model.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 10),
        )

    model.load_state_dict(state_dict)
    # Evaluate the source accuracy 
    seed=0

    # Set the seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create the logger 
    log_dir = os.path.join(os.path.dirname(__file__), '../..', f'logs/ntl/{seed}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, f'ntl_{source_domain}_to_{target_domain}.log')
    # The log file should clear every time
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO)
    # Log the time 
    logging.info(time.asctime(time.localtime(time.time())))
    # Log the args
    # Log the source and target domain
    logging.info(f'NTL: {source_domain} to {target_domain}')

    # Load the dataset
    if source_domain == 'mnist':
        source_trainloader, source_testloader = get_mnist_dataloader(args, ratio=1)
    elif source_domain == 'cifar10':
        source_trainloader, source_testloader = get_cifar_dataloader(args, ratio=1)
    elif source_domain == 'usps':
        source_trainloader, source_testloader = get_usps_dataloader(args, ratio=1)
    elif source_domain == 'svhn':
        source_trainloader, source_testloader = get_svhn_dataloader(args, ratio=1)
    elif source_domain == 'mnistm':
        source_trainloader, source_testloader = get_mnistm_dataloader(args, ratio=1)
    elif source_domain == 'syn':
        source_trainloader, source_testloader = get_syn_dataloader(args, ratio=1)
    elif source_domain == 'stl':
        source_trainloader, source_testloader = get_stl_dataloader(args, ratio=1)

    
    # Load the target dataset
    if target_domain == 'mnist':
        target_trainloader, target_testloader = get_mnist_dataloader(args, ratio=1)
    elif target_domain == 'cifar10':
        target_trainloader, target_testloader = get_cifar_dataloader(args, ratio=1)
    elif target_domain == 'usps':
        target_trainloader, target_testloader = get_usps_dataloader(args, ratio=1)
    elif target_domain == 'svhn':
        target_trainloader, target_testloader = get_svhn_dataloader(args, ratio=1)
    elif target_domain == 'mnistm':
        target_trainloader, target_testloader = get_mnistm_dataloader(args, ratio=1)
    elif target_domain == 'syn':
        target_trainloader, target_testloader = get_syn_dataloader(args, ratio=1)
    elif target_domain == 'stl':
        target_trainloader, target_testloader = get_stl_dataloader(args, ratio=1)

    # Evaluate the model
    print("Evaluate the model on source domain")
    model_copy = deepcopy(model)
    finetune_model(model_copy, source_trainloader, source_testloader, lr=1e-4)
    best_acc = evaluate_model(model_copy, source_testloader)
    logging.info(f'NTL: {source_domain} to {target_domain} dataset, the SOURCE DOMAIN best accuracy is {best_acc}')

    # Evaluate the model transferability with different number of fine-tuning samples
    total_train_samples = len(target_trainloader.dataset)

    # Training sample number
    ratios = np.array([0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    train_sample_nums = [int(total_train_samples*ratio) for ratio in ratios]

    for train_sample_num, ratio in zip(train_sample_nums, ratios):
        print(f"Train sample num: {train_sample_num}")
        # Randomly select the samples
        indices = np.random.choice(total_train_samples, train_sample_num, replace=False)
        train_subset = torch.utils.data.Subset(target_trainloader.dataset, indices)
        subtrainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
        # Evaluate the transferability 
        model_copy = deepcopy(model)

        finetune_model(model_copy, subtrainloader, target_testloader, lr=1e-4)
        best_acc = evaluate_model(model_copy, target_testloader)
        logging.info(f'Data ratio {ratio}, Data volume {train_sample_num},  CUTI transfer from {source_domain} to {target_domain} dataset, the best accuracy is {best_acc}')


    pass

if __name__ == '__main__':
    main()