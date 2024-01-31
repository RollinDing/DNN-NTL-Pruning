"""
Measure the similarity between source domain and target domain
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
from utils.args import get_args
from utils.data import *

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def mmd_on_input_space(source_loader, target_loader):
    """
    Measure the dataset similarity in the input feature space (using MMD)
    """
    mmd_loss_total = 0.0
    # Select first 10 batches 
    batch_num = 10
    for idx, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):
        if idx >= batch_num:
            break
        source, _ = source_data
        target, _ = target_data
        source = source.view(source.size(0), -1)
        target = target.view(target.size(0), -1)

        # make sure source and target have the same number of samples
        min_len = min(len(source), len(target))
        source = source[:min_len]
        target = target[:min_len]

        mmd_loss = mmd(source, target)
        mmd_loss_total += mmd_loss.item()

    mmd_loss = mmd_loss_total / batch_num
    return mmd_loss

def conditional_mmd_on_input_space(source_loader, target_loader, class_num=10):
    """
    Measure the dataset similarity in the input feature space (using conditional MMD)
    Compute the MMD conditional on the labels
    """
    mmd_loss_total = [0]*class_num
    # Select first 10 batches
    batch_num = 10
    for idx, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):
        if idx >= batch_num:
            break
        for classidx in range(class_num):
            source, source_labels = source_data
            target, target_labels = target_data
            source = source.view(source.size(0), -1)
            target = target.view(target.size(0), -1)
            source = source[source_labels == classidx]
            target = target[target_labels == classidx]
            
            # make sure source and target have the same number of samples
            min_len = min(len(source), len(target))
            source = source[:min_len]
            target = target[:min_len]

            mmd_loss = mmd(source, target)
            mmd_loss_total[classidx] += mmd_loss.item()

    # The total loss is the average of mmd loss
    mmd_loss_total = sum(mmd_loss_total)/class_num/batch_num
    return mmd_loss_total

def mmd_on_feature_space(encoder, mask_dict, source_loader, target_loader, device):
    """
    Measure the dataset similarity in the feature space (using MMD)
    """
    mmd_loss_total = 0.0


    # Select first 10 batches 
    batch_num = 10
    for idx, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):
        if idx >= batch_num:
            break
        source, _ = source_data
        target, _ = target_data
        source = source.to(device)
        target = target.to(device)
        
        source = encoder(source, mask_dict).view(source.size(0), -1)
        target = encoder(target, mask_dict).view(target.size(0), -1)

        # make sure source and target have the same number of samples
        min_len = min(len(source), len(target))
        source = source[:min_len]   
        target = target[:min_len]

        mmd_loss = mmd(source, target)
        mmd_loss_total += mmd_loss.item()

    mmd_loss = mmd_loss_total/batch_num
    return mmd_loss

def conditional_mmd_on_feature_space(encoder, mask_dict, source_loader, target_loader, device, class_num=10):
    """
    Measure the dataset similarity in the feature space (using conditional MMD)
    Compute the MMD conditional on the labels
    """
    mmd_loss_total = [0]*class_num
    # Select first 10 batches
    batch_num = 10
    for idx, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):
        if idx >= batch_num:
            break
        for classidx in range(class_num):
            source, source_labels = source_data
            target, target_labels = target_data
            source = source.to(device)
            target = target.to(device)
            source = encoder(source, mask_dict).view(source.size(0), -1)
            target = encoder(target, mask_dict).view(target.size(0), -1)
            source = source[source_labels == classidx]
            target = target[target_labels == classidx]
            
            # make sure source and target have the same number of samples
            min_len = min(len(source), len(target))
            source = source[:min_len]
            target = target[:min_len]

            mmd_loss = mmd(source, target)
            mmd_loss_total[classidx] += mmd_loss.item()

    mmd_loss_total = sum(mmd_loss_total)/class_num/batch_num
    return mmd_loss_total

def finetune_sparse_encoder(encoder, classifier, mask_dict, trainloader, testloader, nepochs=50, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    classifier.to(device)

    # fine-tune the encoder and classifier
    # optimizer = torch.optim.SGD([param for name, param in encoder.named_parameters() if param.requires_grad] + [param for name, param in classifier.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
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
                    # param.grad = param.grad * mask_dict[name]

        print(f"Epoch {epoch}: {total_loss/count}")

        # # how many percentage parameters are adjusted 
        # changed = 0
        # total = 0
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         changed += torch.sum(param.grad != 0).item()
        #         total += param.numel()
        # print(f"Percentage of changed parameters: {changed/total}")

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
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)

    source_domain = args.source
    target_domain = args.target
    finetune_ratio = args.finetune_ratio

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

    model = load_base_model(model, args.arch, source_domain, source_trainloader, source_testloader)

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
    
    resnet_encoder = ResNetEncoder(model)
    resnet_classifier = ResNetClassifier(model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Measure the dataset similarity in the input feature space (using MMD)
    mmd_loss = mmd_on_input_space(source_trainloader, target_trainloader)
    print(f"MMD on input space from source {args.source} to target {args.target} is {mmd_loss}")

    mmd_loss = conditional_mmd_on_input_space(source_trainloader, target_trainloader)
    print(f"Conditional MMD on input space from source {args.source} to target {args.target} is {mmd_loss}")

    mask_dict = {}
    for name, module in resnet_encoder.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask_dict[name] = torch.ones(module.weight.shape).to(device)

    reference = 'usps'
    mask_path  = f'saved_models/{args.arch}/{source_domain}_to_{reference}/admm_mask.pth'
    encoder_path = f'saved_models/{args.arch}/{source_domain}_to_{reference}/admm_encoder.pth'
    classifier_path = f'saved_models/{args.arch}/{source_domain}_to_{reference}/admm_source_classifier.pth'
    
    mask_dict = torch.load(mask_path)
    resnet_encoder = torch.load(encoder_path)
    resnet_classifier = torch.load(classifier_path)

    
    # Measure the dataset similarity in the feature space (using MMD)

    resnet_encoder = resnet_encoder.to(device)
    mmd_loss = mmd_on_feature_space(resnet_encoder, mask_dict, source_trainloader, target_trainloader, device)
    print(f"MMD on feature space from source {args.source} to target {args.target} is {mmd_loss}")

    mmd_loss = conditional_mmd_on_feature_space(resnet_encoder, mask_dict, source_trainloader, target_trainloader, device)
    print(f"Conditional MMD on feature space from source {args.source} to target {args.target} is {mmd_loss}")

    # Test the transferability 

    finetune_sparse_encoder(resnet_encoder, resnet_classifier, mask_dict, target_trainloader, target_testloader, nepochs=50, lr=0.001)
    evaluate_sparse_encoder(resnet_encoder, resnet_classifier, mask_dict, target_testloader)
    # pass 

if __name__ == "__main__":
    # Set the random seed for reproducible experiments
    torch.manual_seed(1234)
    np.random.seed(1234)

    main()