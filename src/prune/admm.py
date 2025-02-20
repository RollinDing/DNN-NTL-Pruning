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
import pickle
from copy import deepcopy
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vgg import PrunableVGG, PrunableResNet18

from prune.pruner import load_base_model
from utils.args import get_args
from utils.data import *

class ADMMPruner:
    def __init__(self, model, source_loader, target_loader, args, prune_percentage=0.9, source_perf_threshold=0.9, max_iterations=30):    
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
                new_z[abs(z)<=delta] = 0
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
    
    def update_weights(self, Z_dict, U_dict, rho, alpha):
        # Update model weights using ADMM loss
        loss_sum = 0
        admm_loss_sum = 0
        target_loss_sum = 0
        sample_num = 0
        # Initialize the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        # Initialize the loss function
        criterion = nn.CrossEntropyLoss()

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
                # loss = source_loss - alpha*torch.clamp(target_loss, max=10)
                loss = source_loss + torch.log(1 + alpha*source_loss/target_loss)
                # The admm loss is the loss + rho/2 * sum((param - Z + U)^2)

                # Compute ADMM regularization term with detached Z and U
                admm_reg = sum([torch.norm((param - Z_dict[name].detach() + U_dict[name].detach()))
                                for name, param in self.model.named_parameters() if param.requires_grad])

                # Compute the total ADMM loss
                admm_loss = loss + rho/2 * admm_reg
                admm_loss.backward()
                optimizer.step()

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

            scheduler.step()
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

def evaluate_sparse_model(model, mask_dict, trainloader, testloader, nepochs=30, lr=0.001):
    # Fine-tune the model using trainloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    # Only fine-tune the unfrozen parameters
    optimizer = torch.optim.SGD([param for name, param in model.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

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
        logging.info(f"Epoch {epoch}: {total_loss/count}")
    
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

    # logging.info(f"Evaluate Accuracy: {correct/total}")
    return correct/total

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

    model = load_base_model(model, args.arch, source_domain, source_trainloader, source_testloader)

    # Show the model architecture
    print("Modify the model with prunable architecture!")
    if args.arch == 'vgg11':
        model2prune = PrunableVGG(model)
    elif args.arch == 'resnet18':
        model2prune = PrunableResNet18(model)

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
    
    modelcopy = deepcopy(model2prune)
    admm_copy = ADMMPruner(modelcopy, source_trainloader, target_trainloader, args, max_iterations=200, prune_percentage=0.98)
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
    admm_pruner = ADMMPruner(model2prune, source_trainloader, target_trainloader, args, max_iterations=100, prune_percentage=0.98)
    
    # Evaluate the model
    # admm_pruner.evaluate(source_testloader)
    # admm_pruner.evaluate(target_testloader)

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
    torch.save(admm_pruner.model, f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_model.pth')
    torch.save(admm_pruner.mask_dict, f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_mask.pth')

    # Fine-tune the model 
    # The model sparsity
    logging.info(f"Model Sparsity: {admm_pruner.model_sparsity()}")
    logging.info("Before Fine-tune the model")
    print("Evaluate one the source domain")
    model = admm_pruner.model
    mask_dict = admm_pruner.mask_dict
    source_model = deepcopy(model)
    source_accuracy = evaluate_sparse_model(source_model, mask_dict, source_trainloader, source_testloader, nepochs=50, lr=1e-4)
    # Save the finetuned source model
    torch.save(source_model, f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_source_model.pth')

    logging.info(f"Source accuracy: {source_accuracy}")

    logging.info("Evaluate on the target domain")
    target_model = deepcopy(model)
    target_accuracy = evaluate_sparse_model(target_model, mask_dict, target_trainloader, target_testloader, nepochs=50, lr=1e-4)

    # Save the finetuned target model
    torch.save(target_model, f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_target_model.pth')
    logging.info(f"Target accuracy: {target_accuracy}")
    pass


if __name__ == "__main__":
    # Set the random seed for reproducible experiments
    torch.manual_seed(1234)
    np.random.seed(1234)

    main()