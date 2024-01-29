"""
Implement the encoder-based pruning method using ADMM for model non-transferability
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


class ADMMEncoderPruner:
    def __init__(self, encoder, classifier, source_loader, target_loader, args, prune_percentage=0.9, source_perf_threshold=0.9, max_iterations=30):    
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.args = args
        self.nepochs = args.epochs
        self.lr = args.lr
        self.prune_percentage = prune_percentage
        self.source_perf_threshold = source_perf_threshold
        self.max_iterations = max_iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # The model including two parts:
            # 1. encoder: the encoder part of the model (Shared by both domains -- sparse and to be pruned)
            # 2. classifier: the classifier part of the model (Domain-specific -- dense and not to be pruned)
        self.encoder = encoder
        self.source_classifier = deepcopy(classifier)
        self.target_classifier = deepcopy(classifier)

        self.mask_dict = {}
        # initialize the mask_dict
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.mask_dict[name] = torch.ones_like(param).to(self.device)
    
    def evaluate(self, data_loader):
        # Evaluation the model with the mask applied
        self.encoder.eval()
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
        self.encoder.train()

        # Only fine-tune the unfrozen parameters
        # optimizer = torch.optim.SGD([param for name, param in self.model.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
        optimizer = torch.optim.Adam([param for name, param in self.model.named_parameters() if param.requires_grad], lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(nepochs):
            total_loss = 0.0
            count = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs, self.mask_dict)
                loss = criterion(outputs, labels)
                loss.backward()
                total_loss += loss.item()
                count += len(labels)
                optimizer.step()
            print(f"Epoch {epoch}: {total_loss/count}")

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
    
    def initialize_Z_and_U(self):
        # Initialize the Z and U variables
        Z_dict = {}
        U_dict = {}
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                Z_dict[name] = param.clone().detach().to(self.device)
                U_dict[name] = torch.zeros_like(param).to(self.device)
        return Z_dict, U_dict
    
    def update_Z_l1(self, U_dict, alpha, rho):
        new_Z = {}
        delta = alpha / rho
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:  # Ensure we only consider trainable parameters
                z = (param + U_dict[name]) * self.mask_dict[name]
                new_z = z.clone()  # Clone to avoid in-place operations
                # Apply soft thresholding
                new_z[z>delta] = z[z>delta] - delta
                new_z[z<-delta] = z[z<-delta] + delta
                new_z[abs(z)<=delta] = 0
                new_Z[name] = new_z
        return new_Z
    
    def update_U(self, U_dict, Z_dict):
        # Update the U variables
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                # compute the U
                U_dict[name] = U_dict[name] + param - Z_dict[name]
                # detach the U from the graph
                U_dict[name] = U_dict[name].detach()
        return U_dict
    
    def update_masks(self, Z_dict):
        # Update the masks
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.mask_dict[name] = (Z_dict[name] != 0).float().to(self.device)
        return self.mask_dict
    
    def update_weights(self, Z_dict, U_dict, rho, alpha):
        # Update model weights using ADMM loss
        loss_sum = 0
        admm_loss_sum = 0
        target_loss_sum = 0
        sample_num = 0
        # Initialize the optimizer TODO: How to update the parameters?
        optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        # Initialize the loss function
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.nepochs):
            for (source_input, source_labels), (target_input, target_labels) in zip(self.source_loader, self.target_loader):
                source_input = source_input.to(self.device)
                source_labels = source_labels.to(self.device)
                target_input = target_input.to(self.device)
                target_labels = target_labels.to(self.device)

                optimizer_encoder.zero_grad()

                # The architecture-specific forward pass 
                # TODO???
                if self.args.arch == 'resnet18':
                    source_features = self.encoder(source_input, self.mask_dict)
                    target_features = self.encoder(target_input, self.mask_dict)
                    
                target_outputs = self.model(target_input, self.mask_dict)

                source_loss = criterion(source_outputs, source_labels)
                target_loss = criterion(target_outputs, target_labels)
                # loss = source_loss - alpha*torch.clamp(target_loss, max=10)
                loss = source_loss + torch.log(1 + alpha*source_loss/target_loss)
                # The admm loss is the loss + rho/2 * sum((param - Z + U)^2)

                # Compute ADMM regularization term with detached Z and U
                admm_reg = sum([torch.norm((param - Z_dict[name].detach() + U_dict[name].detach()))
                                for name, param in self.model.named_parameters() if param.requires_grad])

                # Compute the total ADMM loss
                admm_loss = loss + rho/2 * admm_reg
                admm_loss.backward()
                optimizer_encoder.step()

                # apply the mask to the model
                for name, param in self.model.named_parameters():
                    if name in self.mask_dict:
                        param.data = param.data * self.mask_dict[name]
                        # set the gradient to zero
                        param.grad = param.grad * self.mask_dict[name]

                # Record the admm loss
                loss_sum += loss.item()
                target_loss_sum += target_loss.item()
                admm_loss_sum += rho/2 * admm_reg.item()
                sample_num += source_input.size(0)

            # Print the admm loss
            logging.info(f'Epoch {epoch}: admm loss: {admm_loss_sum / sample_num}; task loss: {loss_sum / sample_num}; target loss: {target_loss_sum / sample_num}')
        
        target_optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        source_optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        for epoch in range(10):
            for target_input, target_labels in self.target_loader:
                target_input = target_input.to(self.device)
                target_labels = target_labels.to(self.device)
                target_optimizer.zero_grad()

                # forward + backward + optimize
                target_outputs = self.model(target_input, self.mask_dict)
                target_loss = criterion(target_outputs, target_labels)

                target_loss.backward()
                target_optimizer.step()
            
            for source_input, source_labels in self.source_loader:
                source_input = source_input.to(self.device)
                source_labels = source_labels.to(self.device)
                source_optimizer.zero_grad()

                # forward + backward + optimize
                source_outputs = self.model(source_input, self.mask_dict)
                source_loss = criterion(source_outputs, source_labels)

                source_loss.backward()
                source_optimizer.step()

    def admm_loss(self, device, model, Z, U, rho, output, target, criterion):
        loss = criterion(output, target)
        for name, param in model.named_parameters():
            u = U[name].to(device)
            z = Z[name].to(device)
            loss += rho / 2 * (param - z + u).norm()
        return loss

    def run_admm(self):
        # Run the ADMM algorithm

        # Initialize the rho
        rho = self.args.rho
        alpha = self.args.alpha
        # Initialize the Z and U variables
        Z_dict, U_dict = self.initialize_Z_and_U()
        
        # Run the ADMM iterations
        # Set model to train mode
        self.model.train()
        for iteration in range(self.max_iterations):
            self.update_weights(Z_dict, U_dict, rho, alpha)
            # Update the Z variables
            l1_alpha = 1e-4
            Z_dict = self.update_Z_l1(U_dict, l1_alpha, rho)
            # show the minimal non-zero value
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"Minimal non-zero value of {name}: {torch.max(param)}")
                    break
            # Update the U variables
            U_dict = self.update_U(U_dict, Z_dict)
            # Update the masks
            self.mask_dict = self.update_masks(Z_dict)
            # Evaluate the model
            source_perf = self.evaluate(self.source_loader)
            target_perf = self.evaluate(self.target_loader)
            logging.info(f'Iteration {iteration}: source perf: {source_perf}, target perf: {target_perf}, model sparsity: {self.model_sparsity()}')
            # Check the model sparsity for each epoch as stop criterion
            sparsity = self.model_sparsity()
            if sparsity > self.prune_percentage:
                break



def main():
    pass

if __name__ == '__main__':
    main()