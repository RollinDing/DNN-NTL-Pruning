"""
ADMM prunning for DNN non-transferable learning
"""
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vgg import PrunableVGG

from pruner import load_base_model
from utils.args import get_args
from utils.data import *

class ADMMPruner:
    def __init__(self, model, source_loader, target_loader, args, prune_percentage=0.1, source_perf_threshold=0.9, max_iterations=30):    
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.args = args
        self.nepochs = args.epochs
        self.lr = args.lr
        self.prune_percentage = prune_percentage
        self.source_perf_threshold = source_perf_threshold
        self.max_iterations = max_iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.mask_dict = {}
        # initialize the mask_dict
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.mask_dict[name] = torch.ones_like(param).to(self.device)
    
    def evaluate(self, data_loader):
        # Evaluation the model with the mask applied
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for input, labels in data_loader:
                input = input.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input, self.mask_dict)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # print(f'Accuracy on dataset: {correct / total}\n')
        return correct / total

    def finetune_model(self, dataloader, nepochs=30, lr=1e-3):
        # Fine-tuning the model on both source and target dataset
        self.model.train()

        # Only fine-tune the unfrozen parameters
        optimizer = torch.optim.SGD([param for name, param in self.model.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(nepochs):
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs, self.mask_dict)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def model_sparsity(self):
        # Compute the sparsity of the model considering the pruning masks
        total = 0
        nonzero = 0
        for name, param in self.model.named_parameters():
            # Check if the mask exists for this parameter
            mask = self.mask_dict.get(name, torch.ones_like(param))
            total += mask.numel()
            nonzero += torch.sum(mask != 0).item()
        return 1 - nonzero / total
    
    def initialize_Z_and_U(self):
        # Initialize the Z and U variables
        Z_dict = {}
        U_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                Z_dict[name] = param.clone().detach().to(self.device)
                U_dict[name] = torch.zeros_like(param).to(self.device)
        return Z_dict, U_dict
    
    def update_Z_l1(self, U_dict, alpha, rho):
        new_Z = {}
        delta = alpha / rho
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Ensure we only consider trainable parameters
                z = (param + U_dict[name]) * self.mask_dict[name]
                new_z = z.clone()  # Clone to avoid in-place operations
                # Apply soft thresholding
                new_z[z>delta] = z[z>delta] - delta
                new_z[z<-delta] = z[z<-delta] + delta
                new_z[abs(z) <= delta] = 0
                new_Z[name] = new_z
        return new_Z
    
    def update_U(self, U_dict, Z_dict):
        # Update the U variables
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # compute the U
                U_dict[name] = U_dict[name] + param - Z_dict[name]
                # detach the U from the graph
                U_dict[name] = U_dict[name].detach()
        return U_dict
    
    def update_masks(self, Z_dict):
        # Update the masks
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.mask_dict[name] = (Z_dict[name] != 0).float().to(self.device)
        return self.mask_dict

    def admm_loss(self, device, model, Z, U, rho, output, target, criterion):
        loss = criterion(output, target)
        for name, param in model.named_parameters():
            u = U[name].to(device)
            z = Z[name].to(device)
            loss += rho / 2 * (param - z + u).norm()
        return loss

    def run_admm(self):
        # Run the ADMM algorithm
        # Initialize the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Initialize the loss function
        criterion = nn.CrossEntropyLoss()
        # Initialize the rho
        rho = self.args.rho
        alpha = self.args.alpha
        # Initialize the Z and U variables
        Z_dict, U_dict = self.initialize_Z_and_U()
        # Run the ADMM iterations
        # Set model to train mode
        self.model.train()

        for iteration in range(self.max_iterations):
            # Update model weights using ADMM loss
            loss_sum = 0
            admm_loss_sum = 0
            sample_num = 0
            for epoch in range(self.nepochs):
                for (source_input, source_labels), (target_input, target_labels) in zip(self.source_loader, self.target_loader):
                    source_input = source_input.to(self.device)
                    source_labels = source_labels.to(self.device)
                    target_input = target_input.to(self.device)
                    target_labels = target_labels.to(self.device)

                    optimizer.zero_grad()

                    # forward + backward + optimize
                    source_outputs = self.model(source_input, self.mask_dict)
                    target_outputs = self.model(target_input, self.mask_dict)

                    source_loss = criterion(source_outputs, source_labels)
                    target_loss = criterion(target_outputs, target_labels)
                    loss = source_loss - torch.clamp(alpha*target_loss, max=1)
                    # The admm loss is the loss + rho/2 * sum((param - Z + U)^2)

                    # Compute ADMM regularization term with detached Z and U
                    admm_reg = sum([torch.norm((param - Z_dict[name].detach() + U_dict[name].detach()))
                                    for name, param in self.model.named_parameters() if param.requires_grad])

                    # Compute the total ADMM loss
                    admm_loss = loss + rho/2 * admm_reg
                    # admm_loss = self.admm_loss(self.device, self.model, Z_dict, U_dict, rho, source_outputs, source_labels, criterion)
                    admm_loss.backward()
                    optimizer.step()

                    # Record the admm loss
                    loss_sum += loss.item()
                    admm_loss_sum += rho/2 * admm_reg.item()
                    sample_num += source_input.size(0)
                # Print the admm loss
                logging.info(f'Epoch {epoch}: admm loss: {admm_loss_sum / sample_num}; task loss: {loss_sum / sample_num}')
                
            # Update the Z variables
            l1_alpha = 1e-4
            Z_dict = self.update_Z_l1(U_dict, l1_alpha, rho)
            # Update the U variables
            U_dict = self.update_U(U_dict, Z_dict)
            # Update the masks
            self.mask_dict = self.update_masks(Z_dict)
            # Compute the model sparsity
            # sparsity = self.model_sparsity()
            # Evaluate the model
            source_perf = self.evaluate(self.source_loader)
            target_perf = self.evaluate(self.target_loader)
            logging.info(f'Iteration {iteration}: source perf: {source_perf}, target perf: {target_perf}, model sparsity: {self.model_sparsity()}')


def main():
    # load args 
    args = get_args()

    if args.arch == 'vgg11':
        # Load the pretrained model 
        model = torchvision.models.vgg11(pretrained=True)
        # change the output layer to 10 classes (for digits dataset)
        model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )


    source_domain = args.source
    target_domain = args.target
    finetune_ratio = args.finetune_ratio

    # Create the logger 
    log_dir = os.path.join(os.path.dirname(__file__), '../..', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f'admm_{source_domain}_to_{target_domain}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
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

    model = load_base_model(model, 'vgg11', source_domain, source_trainloader, source_testloader)

    # Show the model architecture
    model2prune = PrunableVGG(model)

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
    
    modelcopy = deepcopy(model2prune)
    admm_copy = ADMMPruner(modelcopy, source_trainloader, target_trainloader, args)
    logging.info("Evaluate the model before ADMM")
    logging.info("The model performance on source domain")
    admm_copy.finetune_model(source_trainloader, lr=1e-4, nepochs=50)
    source_accuracy = admm_copy.evaluate(source_testloader)
    logging.info(f"Source accuracy: {source_accuracy}")
    admm_copy.finetune_model(target_trainloader, lr=1e-4, nepochs=50)
    logging.info("The model performance on target domain")
    target_accuracy = admm_copy.evaluate(target_testloader)
    logging.info(f"Target accuracy: {target_accuracy}")

    # Initialize the ADMM pruner
    admm_pruner = ADMMPruner(model2prune, source_trainloader, target_trainloader, args)
    # Evaluate the model
    # admm_pruner.evaluate(source_testloader)
    # admm_pruner.evaluate(target_testloader)

    # Run the ADMM algorithm
    admm_pruner.run_admm()

    # Fine-tune the model 
    # The model sparsity
    logging.info(f"Model Sparsity: {admm_pruner.model_sparsity()}")
    logging.info("Before Fine-tune the model")
    logging.info("Evaluate one the source domain")
    source_accuracy = admm_pruner.evaluate(source_testloader)
    logging.info(f"Source accuracy: {source_accuracy}")

    logging.info("Evaluate on the target domain")
    target_accuracy = admm_pruner.evaluate(target_testloader)
    logging.info(f"Target accuracy: {target_accuracy}")

    logging.info("Fine-tune the model")
    admm_pruner.finetune_model(target_trainloader, lr=1e-4, nepochs=50)
    logging.info("Evaluate on the target domain")
    target_accuracy = admm_pruner.evaluate(target_testloader)
    logging.info(f"Target accuracy: {target_accuracy}")
    pass


if __name__ == "__main__":
    # Set the random seed for reproducible experiments
    torch.manual_seed(1234)
    np.random.seed(1234)

    main()