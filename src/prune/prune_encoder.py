import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.args import get_args
from prune.pruner import load_base_model
from utils.data import *

from models.vgg import PrunableVGG
from models.encoders import ResNetEncoder, ResNetClassifier
from evaluate.evaluate_encoder_impact import finetune_sparse_encoder, evaluate_sparse_encoder

import matplotlib.pyplot as plt
from tqdm import tqdm as tdqm
from copy import deepcopy
import random
import logging
import time

class ORGEncoderPruner:
    """
    The pruning will only be applied on the original dataset
    """
    def __init__(self, encoder, classifier, dataloader, prune_method='l1_unstructured', prune_percentage=0.1, source_perf_threshold=0.9, max_iterations=10):    
        self.dataloader = dataloader
        self.nepochs = 30
        self.lr = 0.001
        self.prune_method = prune_method
        self.prune_percentage = prune_percentage
        self.source_perf_threshold = source_perf_threshold
        self.max_iterations = max_iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.classifier = classifier
        # Move the model to the device
        self.encoder.to(self.device)
        self.classifier.to(self.device) 

        self.mask_dict = {}
        # initialize the mask_dict
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.mask_dict[name] = torch.ones_like(param).to(self.device)
        
    def evaluate(self, data_loader):
        # Evaluation the model with the mask applied
        self.encoder.train()
        total = 0
        correct = 0
        with torch.no_grad():
            for input, labels in data_loader:
                input = input.to(self.device)
                labels = labels.to(self.device)
                features = self.encoder(input, self.mask_dict)
                outputs = self.classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy on dataset: {correct / total}')
        return correct / total

    def finetune_model(self, dataloader, nepochs=30, lr=1e-3, weight_decay=0.0008):
        # Fine-tuning the model on both source and target dataset
        self.encoder.train()

        # Only fine-tune the unfrozen parameters
        # optimizer = torch.optim.SGD([param for name, param in self.model.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
        optimizer = torch.optim.Adam([param for name, param in self.encoder.named_parameters() if param.requires_grad]+
                                     [param for name, param in self.classifier.named_parameters() if param.requires_grad], lr=lr, weight_decay=weight_decay)
        
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(nepochs):
            total_loss = 0.0
            count = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                features = self.encoder(inputs, self.mask_dict)
                outputs = self.classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                total_loss += loss.item()
                count += len(labels)
                optimizer.step()
            # print(f"Epoch {epoch}: {total_loss/count}")
        
    def compute_gradient_importance(self):
        """
        The importance score is not correct here --> we are going to find the important score to the source domain but the not for the target domain,
        The design of the score should be based on two SEPERATE loss function --> rather than one.
        """
        self.encoder.train()
        model_weights = [param for name, param in self.encoder.named_parameters() if param.requires_grad]
        gradients = self.compute_loader_gradients()

        # The importance scores are the gradients times the weights
        importance_scores = [torch.abs(model_weights[i]) for i, name in enumerate(gradients)]
        # importance_scores = [torch.abs(gradients[name]) for i, name in enumerate(gradients)]

        # the importance score is the weights magnitude
        # importance_scores = [torch.abs(model_weights[i]) for i, name in enumerate(gradients)]
        return importance_scores

    def compute_loader_gradients(self):
        gradients = {name: torch.zeros_like(param) for name, param in self.encoder.named_parameters() if param.requires_grad}
        criterion = torch.nn.CrossEntropyLoss()
        
        self.encoder.train()
        self.classifier.train()
        for inputs, labels in self.dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            features = self.encoder(inputs, self.mask_dict)
            outputs = self.classifier(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            self.encoder.zero_grad()  # Reset gradients to zero
            loss.backward()

            # Accumulate gradients
            for name, param in self.encoder.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Due to cross-entropy loss, the last layer will not have gradients
                    gradients[name] += param.grad

        # Average gradients over the dataset
        num_batches = len(self.dataloader)
        for name in gradients:
            gradients[name] /= num_batches

        return gradients
    
    def prune(self, pruning_ratio):
        """
        Prune the model based on the calculated importance scores.

        Args:
        pruning_ratio (float): The ratio of weights to prune (0.0 to 1.0).
        """
        print('Pruning the model with pruning ratio:', pruning_ratio)
        # Calculate importance scores
        self.importance_scores = self.compute_gradient_importance()
        importance_scores = self.importance_scores

        # Flatten the importance scores and sort them
        all_scores = torch.cat([scores.flatten() for scores in importance_scores])
        threshold_idx = int(pruning_ratio * all_scores.numel())

        # Determine the pruning threshold
        threshold, _ = torch.kthvalue(all_scores, threshold_idx)

        # Apply the threshold to the importance scores and create a mask
        for i, (name, param) in enumerate(self.encoder.named_parameters()):
            if param.requires_grad and param.grad is not None:
                # Compute mask based on the importance score threshold
                mask = importance_scores[i] > threshold
                self.mask_dict[name] = mask.float()
                # print(f'Layer {i} sparsity: {1 - torch.sum(mask).item() / mask.numel()}')

                # Apply the mask to the weights and freeze pruned weights
                with torch.no_grad():
                    param.data.mul_(mask.float())

        # Measure the sparsity of the model
        sparsity = self.model_sparsity()
        print(f'Model sparsity: {sparsity}')
        return self.mask_dict
    
    def model_sparsity(self):
        # Compute the sparsity of the model considering the pruning masks
        total = 0
        nonzero = 0
        for name, param in self.encoder.named_parameters():
            # Check if the mask exists for this parameter
            mask = self.mask_dict.get(name, torch.ones_like(param))
            total += mask.numel()
            nonzero += torch.sum(mask != 0).item()
        return 1 - nonzero / total
    
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

    source_domain = args.source
    target_domain = args.target
    finetune_ratio = args.finetune_ratio

    # Create the logger 
    log_dir = os.path.join(os.path.dirname(__file__), '../..', f'logs/{args.arch}/{args.prune_method}/{args.seed}')
    print("Creating log directory: ", log_dir)
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
    logging.info(f'Original Pruning: {source_domain} to {target_domain}')

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
    
    resnet_encoder = ResNetEncoder(model)
    resnet_classifier = ResNetClassifier(model)

    # Prune the model's encoder with different pruning ratios
    prune_ratios = [0.01, 0.1, 0.5, 0.9, 0.95, 0.98, 0.99]
    for prune_ratio in prune_ratios:
        logging.info(f'Pruning ratio: {prune_ratio}')
        pruner = ORGEncoderPruner(resnet_encoder, resnet_classifier, source_trainloader, prune_method=args.prune_method, prune_percentage=args.percent)
        pruner.prune(prune_ratio)

        # Finetune with the target dataset
        total_train_samples = len(target_trainloader.dataset)
        target_ratios = np.array([0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
        train_sample_nums = [int(total_train_samples*ratio) for ratio in target_ratios]

        # Get the mask 
        mask_dict = pruner.mask_dict
        # Compute the model sparsity using the mask_dict
        total_params = 0
        total_nonzero_params = 0
        for name, param in resnet_encoder.named_parameters():
            total_params += param.numel()
            if name in mask_dict:
                total_nonzero_params += torch.sum(mask_dict[name]).item()
        logging.info(f"Model sparsity: {1-total_nonzero_params/total_params}")

        finetune_sparse_encoder(pruner.encoder, pruner.classifier, mask_dict, source_trainloader, source_testloader, lr=1e-4)
        best_acc = evaluate_sparse_encoder(pruner.encoder, pruner.classifier, mask_dict, source_testloader)
        logging.info(f'Source accuracy: {best_acc}')

        for train_sample_num, ratio in zip(train_sample_nums, target_ratios):
            # Get the pretrained model 
            encoder_copy = deepcopy(pruner.encoder)
            classifier_copy = deepcopy(pruner.classifier)
            print(f"Train sample num: {train_sample_num}")
            # Randomly select the samples
            indices = np.random.choice(total_train_samples, train_sample_num, replace=False)
            train_subset = torch.utils.data.Subset(target_trainloader.dataset, indices)
            subtrainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
            finetune_sparse_encoder(encoder_copy, classifier_copy, mask_dict, subtrainloader, target_testloader, lr=1e-4)
            best_acc = evaluate_sparse_encoder(encoder_copy, classifier_copy, mask_dict, target_testloader)
            logging.info(f'Data ratio {ratio}, Data volume {train_sample_num},  Original pruning transfer from {source_domain} to {target_domain} dataset, the best accuracy is {best_acc}')

    

if __name__ == '__main__':
    main()