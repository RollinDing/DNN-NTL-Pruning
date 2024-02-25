"""
Implementation of model non-transferable pruning algorithm on self supervised models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import models 

import logging
import time
import pickle
from copy import deepcopy
import numpy as np
import sys
import os
import timm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vgg import PrunableVGG, PrunableResNet18
from models.encoders import ResNetEncoder, ResNetClassifier

from prune.pruner import load_base_model
from prune.admm_encoder import ADMMEncoderPruner
from utils.args import get_args
from utils.data import *
import random


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in torchvision.models.resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.f(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return out


def load_ssl_model_weights(model_name):
    """
    Load the SSL model weights into a torchvision model.
    """
    if model_name == 'simclr':
        path = f'base_models/resnet50-{model_name}-cifar10.pth'
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model = Model()
        model.load_state_dict(state_dict, strict=False)
        # Compare the dicts
        compare_state_dicts(model.state_dict(), state_dict)
    else:
        # Define path to your SSL model weights
        path = f'base_models/ssl_models/{model_name}.pth'
        
        # Load the SSL model weights
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Add renaming logic here if necessary, similar to your rename and remove_keys functions
        
        # Load a ResNet-50 model
        model = models.resnet50(pretrained=False)
        
        # Replace the final layer for CIFAR10
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # CIFAR10 has 10 classes
        
        # Load the modified state dict
        model.load_state_dict(state_dict, strict=False)
    return model

def load_ssl_model(args, model, device):
    # load the pretrained model trained with self-supervised learning
    method = "simclr"
    # model_path = f"base_models/{args.arch}-{method}-{args.source}.tar"
    model_path = f"base_models/ssl_models/moco-v1.pth" 
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint
    print(state_dict.keys())
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    model.load_state_dict(state_dict)
    
    return model

def finetune_sparse_encoder(encoder, classifier, mask_dict, trainloader, testloader, nepochs=100, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    classifier.to(device)

    # fine-tune the encoder and classifier
    # optimizer = torch.optim.SGD([param for name, param in classifier.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam([param for name, param in encoder.named_parameters() if param.requires_grad] + [param for name, param in classifier.named_parameters() if param.requires_grad], lr=lr, weight_decay=0.0008)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nepochs)
    criterion = torch.nn.CrossEntropyLoss()

    encoder.train()
    classifier.train()
    for epoch in range(nepochs):
        print("Epoch: ", epoch)
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
        print(f"Training Loss: {total_loss/count}")
        # Evaluate the model for every epoch 
        evaluate_sparse_encoder(encoder, classifier, mask_dict, testloader)
        scheduler.step()
        # print current learning rate
        print(f"Current learning rate: {scheduler.get_last_lr()}")

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
    return correct/total

def compare_state_dicts(model_state_dict, loaded_state_dict):
    model_keys = set(model_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())

    missing_keys = model_keys - loaded_keys
    unexpected_keys = loaded_keys - model_keys

    if missing_keys:
        print("Missing keys in the loaded state dict:", missing_keys)
    else:
        print("No missing keys in the loaded state dict.")

    if unexpected_keys:
        print("Unexpected keys in the loaded state dict:", unexpected_keys)
    else:
        print("No unexpected keys in the loaded state dict.")

def main():
    # load args 
    args = get_args()
    # Set the global random seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
    model_name = args.model_name

    # Create the logger 
    log_dir = os.path.join(os.path.dirname(__file__), '../..', f'logs/{args.arch}-{model_name}/{args.seed}/')
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
    # model = load_ssl_model(args, model, device)
    model = load_ssl_model_weights(model_name)
    if not model:
        raise ValueError("Model not found.")
    model.to(device)
    if model_name == 'simclr':

        pass
    else:
        loaded_state_dict = torch.load(f'base_models/ssl_models/{model_name}.pth', map_location='cpu')

        # If your loaded state dict is nested under 'state_dict' key, adjust as necessary
        if 'state_dict' in loaded_state_dict:
            loaded_state_dict = loaded_state_dict['state_dict']
            # Compare the dicts
            compare_state_dicts(model.state_dict(), loaded_state_dict)

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
    resnet_classifier = ResNetClassifier(torchvision.models.resnet50(pretrained=False), num_classes=num_classes)
    # Initialize the original mask is all ones 
    mask_dict = {}
    # initialize the mask_dict
    for name, param in resnet_encoder.named_parameters():
        if param.requires_grad:
            mask_dict[name] = torch.ones_like(param)

    # admm_pruner.finetune_model(source_trainloader)
    # # admm_pruner.initialize_target_classifier()
    # # Evaluate the model
    # admm_pruner.evaluate(source_testloader)
    # admm_pruner.evaluate(target_testloader)
    
    ssl_model_path = os.path.join(os.path.dirname(__file__), '../..', f'saved_models/ssl_models/{args.arch}-{model_name}/{source_domain}_to_{target_domain}')
    if not os.path.exists(ssl_model_path):
        os.makedirs(ssl_model_path)
    
    # If the model is not finetuned, finetune the model
    if not os.path.exists(ssl_model_path + '/encoder.pth'):
        # finetune the encoder and classifier
        finetune_sparse_encoder(resnet_encoder, resnet_classifier, mask_dict, source_trainloader, source_testloader, nepochs=100, lr=1e-4)
        # save the pretrained resnet encoder and resnet classifier
        torch.save(resnet_encoder, ssl_model_path + '/encoder.pth')
        torch.save(resnet_classifier, ssl_model_path + '/classifier.pth')
        torch.save(mask_dict, ssl_model_path + '/mask.pth')
    else:
        resnet_encoder = torch.load(ssl_model_path + '/encoder.pth')
        resnet_classifier = torch.load(ssl_model_path + '/classifier.pth')
        mask_dict = torch.load(ssl_model_path + '/mask.pth')

    print(resnet_classifier)
    # save the pretrained resnet encoder and resnet classifier
    # Evaluate the model
    evaluate_sparse_encoder(resnet_encoder, resnet_classifier, mask_dict, source_testloader)    
    admm_pruner = ADMMEncoderPruner(resnet_encoder, resnet_classifier, source_trainloader, target_trainloader, args, max_iterations=200, prune_percentage=args.sparsity)
    
    # admm_pruner.finetune_model(source_trainloader, nepochs=1, lr=1e-5)
    
    admm_pruner.initialize_target_classifier()
    admm_pruner.evaluate(source_testloader, target=False)
    admm_pruner.evaluate(target_testloader, target=True)
    
    # Run the ADMM algorithm
    admm_pruner.run_admm()

    # Create the directory to save the model
    model_dir = os.path.join(os.path.dirname(__file__), '../..', f'saved_models/{args.arch}/{model_name}/{args.prune_method}/{source_domain}_to_{target_domain}/{args.seed}')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # # save the ADMM pruner as a pickle file
    # with open(f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_pruner.pkl', 'wb') as f:
    #     pickle.dump(admm_pruner, f)

    # Save the pruned model and masks
    torch.save(admm_pruner.encoder, f'saved_models/{args.arch}/{model_name}/{args.prune_method}/{source_domain}_to_{target_domain}/{args.seed}/admm_encoder.pth')
    torch.save(admm_pruner.source_classifier, f'saved_models/{args.arch}/{model_name}/{args.prune_method}/{source_domain}_to_{target_domain}/{args.seed}/admm_source_classifier.pth')
    torch.save(admm_pruner.mask_dict, f'saved_models/{args.arch}/{model_name}/{args.prune_method}/{source_domain}_to_{target_domain}/{args.seed}/admm_mask.pth')

    acc=evaluate_sparse_encoder(admm_pruner.encoder, admm_pruner.source_classifier, admm_pruner.mask_dict, source_testloader) 
    logging.info(f"Source Accuracy: {acc}")   

if __name__ == "__main__":
    main()