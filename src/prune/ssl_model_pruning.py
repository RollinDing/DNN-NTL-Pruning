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
from models.encoders import ResNetEncoder, ResNetClassifier

from prune.pruner import load_base_model
from prune.admm_encoder import ADMMEncoderPruner
from utils.args import get_args
from utils.data import *


def load_ssl_model(args, model, device):
    # load the pretrained model trained with self-supervised learning
    method = "simclr"
    # model_path = f"base_models/{args.arch}-{method}-{args.source}.tar"
    model_path = f"base_models/ssl_models/moco-v1.pth" 
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    
    return model

def finetune_sparse_encoder(encoder, classifier, mask_dict, trainloader, testloader, nepochs=30, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    classifier.to(device)

    # fine-tune the encoder and classifier
    # optimizer = torch.optim.SGD([param for name, param in classifier.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam([param for name, param in encoder.named_parameters() if param.requires_grad] + [param for name, param in classifier.named_parameters() if param.requires_grad], lr=lr, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss()

    encoder.train()
    classifier.train()
    for epoch in range(nepochs):
        total_loss = 0.0
        count = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            features = encoder(inputs, mask_dict)
            outputs  = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            count += len(labels)
            optimizer.step()

            # apply the mask to the model
            for name, param in encoder.named_parameters():
                if name in mask_dict:
                    param.data = param.data * mask_dict[name]
                    # set the gradient to zero
                    param.grad = param.grad * mask_dict[name]

        print(f"Epoch {epoch}: {total_loss/count}")

def evaluate_sparse_encoder(encoder, classifier, mask_dict, testloader):
    # Fine-tune the model using trainloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    classifier.to(device)

    # Evaluate the model using testloader
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            features = encoder(inputs, mask_dict)
            outputs = classifier(features)
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
    elif args.arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
    elif args.arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_classes)

    source_domain = args.source
    target_domain = args.target
    finetune_ratio = args.finetune_ratio

    # Create the logger 
    log_dir = os.path.join(os.path.dirname(__file__), '../..', f'logs/{args.arch}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, f'admm_{source_domain}_to_{target_domain}.log')
    # The log file should clear every time
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO)
    # Log the time 
    logging.info(time.asctime(time.localtime(time.time())))
    # Log the args
    logging.info(args)
    # Log the source and target domain
    logging.info(f'ADMM: {source_domain} to {target_domain}')

    # Load the source dataset
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
    elif source_domain == 'imagenette':
        source_trainloader, source_testloader = get_imagenette_dataloader(args, ratio=finetune_ratio)
    elif source_domain == 'imagewoof':
        source_trainloader, source_testloader = get_imagewoof_dataloader(args, ratio=finetune_ratio)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_ssl_model(args, model, device)

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
    elif target_domain == 'imagenette':
        target_trainloader, target_testloader = get_imagenette_dataloader(args, ratio=finetune_ratio)
    elif target_domain == 'imagewoof':
        target_trainloader, target_testloader = get_imagewoof_dataloader(args, ratio=finetune_ratio)
    
    resnet_encoder = ResNetEncoder(model)
    resnet_classifier = ResNetClassifier(model)
    # Initialize the ADMM pruner
    admm_pruner = ADMMEncoderPruner(resnet_encoder, resnet_classifier, source_trainloader, target_trainloader, args, max_iterations=200, prune_percentage=0.98)
    mask_dict = admm_pruner.mask_dict
    # admm_pruner.finetune_model(source_trainloader)
    # # admm_pruner.initialize_target_classifier()
    # # Evaluate the model
    # admm_pruner.evaluate(source_testloader)
    # admm_pruner.evaluate(target_testloader)
    
    # finetune the encoder and classifier
    finetune_sparse_encoder(resnet_encoder, resnet_classifier, mask_dict, source_trainloader, source_testloader, nepochs=30, lr=1e-4)
    # Evaluate the model
    evaluate_sparse_encoder(resnet_encoder, resnet_classifier, mask_dict, source_testloader)    
    exit()
    # Run the ADMM algorithm
    admm_pruner.run_admm()

    # Create the directory to save the model
    model_dir = os.path.join(os.path.dirname(__file__), '../..', f'saved_models/{args.arch}/{source_domain}_to_{target_domain}')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save the ADMM pruner as a pickle file
    with open(f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_pruner.pkl', 'wb') as f:
        pickle.dump(admm_pruner, f)

    # Save the pruned model and masks
    torch.save(admm_pruner.encoder, f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_encoder.pth')
    torch.save(admm_pruner.source_classifier, f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_source_classifier.pth')
    torch.save(admm_pruner.mask_dict, f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_mask.pth')


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    main()