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

from prune.ssl_model_pruning import load_ssl_model, load_ssl_model_weights
from models.encoders import ResNetEncoder, ResNetClassifier
from prune.admm_encoder import ADMMEncoderPruner

from utils.args import get_args
from utils.data import *

def finetune_sparse_encoder(encoder, classifier, mask_dict, trainloader, testloader, nepochs=30, lr=0.001):
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
        for n, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            features = encoder(inputs, mask_dict)
            outputs  = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss = loss.item()
            count = len(labels)
            optimizer.step()

            # apply the mask to the model
            for name, param in encoder.named_parameters():
                if name in mask_dict:
                    param.data = param.data * mask_dict[name]
                    # set the gradient to zero
                    param.grad = param.grad * mask_dict[name]
            if n % 10 == 0:
                print(f"Epoch {epoch} Batch {n}: {total_loss/count}")

        # # how many percentage parameters are adjusted 
        # changed = 0
        # total = 0
        # for name, param in encoder.named_parameters():
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
    return correct/total

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
            total_loss = loss.item()
            count = len(labels)
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

        # load args 
    args = get_args()
    # Set random seed 
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
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
        # Load the pretrained model 
        model = torchvision.models.resnet18(pretrained=True)
        # change the output layer to 10 classes (for digits dataset)
        model.fc = nn.Linear(512, num_classes)

    source_domain = args.source
    target_domain = args.target
    finetune_ratio = args.finetune_ratio
    model_name = args.model_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the logger 
    log_dir = os.path.join(os.path.dirname(__file__), '../..', f'logs/ssl/{args.prune_method}/{model_name}/{args.seed}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, f'admm_{source_domain}_to_{target_domain}.log')
    # The log file should clear every time
    logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO)
    # Log the time 
    logging.info(time.asctime(time.localtime(time.time())))
    # Log the args
    logging.info(args)
    # Log the source and target domain
    logging.info(f'ADMM: {source_domain} to {target_domain}')

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

    # model = load_ssl_model(args, model, device)
    model = load_ssl_model_weights(model_name)

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
    resnet_encoder = ResNetEncoder(model)
    if model_name == 'simclr':
        resnet_classifier = ResNetClassifier(torchvision.models.resnet50(pretrained=False), num_classes=10)
    else:
        resnet_classifier = ResNetClassifier(model, num_classes=10)

    if args.prune_method != 'original':
        # Load the pretrained model from saved state dict
        encoder_path = f'saved_models/{args.arch}/{model_name}/{args.prune_method}/{source_domain}_to_{target_domain}/{args.seed}/admm_encoder.pth'
        classifier_path = f'saved_models/{args.arch}/{model_name}/{args.prune_method}/{source_domain}_to_{target_domain}/{args.seed}/admm_source_classifier.pth'
        mask_path  = f'saved_models/{args.arch}/{model_name}/{args.prune_method}/{source_domain}_to_{target_domain}/{args.seed}/admm_mask.pth'
        mask_dict = torch.load(mask_path)
        resnet_encoder = torch.load(encoder_path)
        resnet_classifier = torch.load(classifier_path)
    else:
        resnet_encoder = ResNetEncoder(model)
        # resnet_classifier = ResNetClassifier(model)
        admm_pruner = ADMMEncoderPruner(resnet_encoder, resnet_classifier, source_trainloader, target_trainloader, args, max_iterations=50, prune_percentage=0.98)
        mask_dict = admm_pruner.mask_dict
    


    # admm_pruner = ADMMPruner(pruned_model, source_trainloader, target_trainloader, args)

    # mask_dict = torch.load(mask_path)
    # Compute the model sparsity using the mask_dict
    total_params = 0
    total_nonzero_params = 0
    for name, param in resnet_encoder.named_parameters():
        total_params += param.numel()
        if name in mask_dict:
            total_nonzero_params += torch.sum(mask_dict[name]).item()
    print(f"Model sparsity: {1-total_nonzero_params/total_params}")
    logging.info(f"Model sparsity: {1-total_nonzero_params/total_params}")

    # Evaluate the model
    print("Evaluate the model on source domain")
    source_encoder = deepcopy(resnet_encoder)
    source_classifier = deepcopy(resnet_classifier)
    # finetune_sparse_encoder(source_encoder, source_classifier, mask_dict, source_trainloader, source_testloader, lr=1e-4)
    best_acc = evaluate_sparse_encoder(source_encoder, source_classifier, mask_dict, source_testloader)
    logging.info(f'ADMM on {model_name}: {source_domain} to {target_domain} dataset, the SOURCE DOMAIN best accuracy is {best_acc}')


    print("Evaluate the model on target domain")

    # Evaluate the model transferability with different number of fine-tuning samples
    total_train_samples = len(target_trainloader.dataset)
    print(total_train_samples)

    # Training sample number
    ratios = np.array([0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    train_sample_nums = [int(total_train_samples*ratio) for ratio in ratios]
    
    # build an all-one mask 
    # all_one_mask_dict = {}
    # for name, param in target_encoder.named_parameters():
    #     if name in mask_dict:
    #         all_one_mask_dict[name] = torch.ones_like(mask_dict[name])

    # # Replace the weights in target model with the remain weights in source model
    # for name, param in source_model.named_parameters():
    #     if name in mask_dict:
    #         target_model.state_dict()[name].data = param.data * mask_dict[name] + target_model.state_dict()[name].data * (1 - mask_dict[name])

    # For each train sample num, randomly select the samples and evaluate the transferability
    for train_sample_num, ratio in zip(train_sample_nums, ratios):
        target_encoder = deepcopy(resnet_encoder)
        target_classifier = deepcopy(resnet_classifier)
        target_encoder.to(device)
        target_classifier.to(device)
        print(f"Train sample num: {train_sample_num}")
        # Randomly select the samples
        indices = np.random.choice(total_train_samples, train_sample_num, replace=False)
        train_subset = torch.utils.data.Subset(target_trainloader.dataset, indices)
        subtrainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
        # Evaluate the transferability 
        encoder_copy = deepcopy(target_encoder)    

        finetune_sparse_encoder(encoder_copy, target_classifier, mask_dict, subtrainloader, target_testloader, lr=1e-4)
        best_acc = evaluate_sparse_encoder(encoder_copy, target_classifier, mask_dict, target_testloader)
        logging.info(f'Data ratio {ratio}, Data volume {train_sample_num},  NTL+LDA transfer from {source_domain} to {target_domain} dataset, the best accuracy is {best_acc}')


if __name__ == "__main__":
    # Set the random seed for reproducible experiments
    main()
