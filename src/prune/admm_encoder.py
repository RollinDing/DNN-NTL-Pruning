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
from models.encoders import ResNetEncoder, ResNetClassifier, VGGEncoder, VGGClassifier

from prune.pruner import load_base_model
from utils.args import get_args
from utils.data import *

import torch
import torch.nn.functional as F
import random

def sfda_multiclass_regularization_loss(outputs, targets, num_classes, data_type, lambda_reg=0.01):
    """
    FDA regularization loss for multi-class classification.
    
    outputs: model predictions, shape (batch_size, num_features)
    targets: ground truth labels, shape (batch_size,)
    num_classes: number of classes in the classification task
    lambda_reg: regularization strength
    """
    device = outputs.device
    class_means = []
    within_class_scatter = 0.0
    
    # Calculate means for each class and within-class scatter
    for class_id in range(num_classes):
        class_indices = targets == class_id
        outputs_class = outputs[class_indices]
        
        if outputs_class.size(0) > 0:  # Check if the class is present in the batch
            class_mean = torch.mean(outputs_class, dim=0)
            within_class_scatter += torch.mean((outputs_class - class_mean).pow(2))
            class_means.append(class_mean)
        else:
            class_means.append(torch.zeros(outputs.shape[1], device=device))
    
    class_means = torch.stack(class_means)  # Shape: (num_classes, num_features)
    
    # Compute total mean for between-class scatter
    total_mean = torch.mean(class_means, dim=0)
    
    # Compute between-class scatter
    between_class_scatter = torch.mean((class_means - total_mean).pow(2))
    
    # Regularization term: Minimize within-class scatter, maximize between-class scatter
    if data_type == 'target':
        reg_loss = lambda_reg * ( between_class_scatter/within_class_scatter)
    elif data_type == 'source':
        reg_loss = lambda_reg * ( within_class_scatter/between_class_scatter)
    return reg_loss

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
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
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


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
        # Move the model to the device
        self.encoder.to(self.device)
        self.source_classifier.to(self.device)
        self.target_classifier.to(self.device)

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
                outputs = self.source_classifier(features)
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
                                     [param for name, param in self.source_classifier.named_parameters() if param.requires_grad], lr=lr, weight_decay=weight_decay)
        
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(nepochs):
            total_loss = 0.0
            count = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                features = self.encoder(inputs, self.mask_dict)
                outputs = self.source_classifier(features)
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
    
    def initialize_target_classifier(self):
        # Initialize the target classifier
        # Initialize the optimizer
        optimizer = optim.Adam([param for name, param in self.target_classifier.named_parameters() if param.requires_grad], lr=1e-3, weight_decay=0.0008)
        # Initialize the loss function
        criterion = nn.CrossEntropyLoss()

        best_loss = 0.0
        patience = 10
        for epoch in range(100):
            total_loss = 0.0
            for target_input, target_labels in self.target_loader:
                target_input = target_input.to(self.device)
                target_labels = target_labels.to(self.device)
                optimizer.zero_grad()

                # forward + backward + optimize
                target_features = self.encoder(target_input, self.mask_dict)
                target_outputs = self.target_classifier(target_features)
                target_loss = criterion(target_outputs, target_labels)

                target_loss.backward()
                optimizer.step()
                total_loss += target_loss.item()

                # apply the mask to the model
                for name, param in self.encoder.named_parameters():
                    if name in self.mask_dict:
                        param.data = param.data * self.mask_dict[name]
                        # set the gradient to zero
                        param.grad = param.grad * self.mask_dict[name]
            
            print(f"Epoch {epoch}: {total_loss/len(self.target_loader)}")
            # Early Stop Criterion:
            # If the loss does not decrease for 10 epochs, stop the training
            if total_loss > best_loss:
                best_loss = total_loss
                patience = 10
            else:
                patience -= 1
                if patience == 0:
                    break

    def update_weights(self, Z_dict, U_dict, rho, alpha):
        # Iteratively update the model weights with ntl, on target dataset and on source dataset
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=0.0008)
        # Build an surrogate encoder to find the real target loss
        target_optimizer = optim.Adam([param for name, param in self.target_classifier.named_parameters() if param.requires_grad], lr=1e-2, weight_decay=0.0008)
        source_optimizer = optim.Adam([param for name, param in self.encoder.named_parameters() if param.requires_grad], lr=1e-4, weight_decay=0.0008)
       
        # Initialize the loss function
        criterion = nn.CrossEntropyLoss()

        all_admm_loss = 0
        nepoch = 10
        for epoch in range(nepoch):
            # Update model weights using ADMM loss
            loss_sum = 0
            admm_loss_sum = 0
            source_loss_sum = 0
            target_loss_sum = 0
            sample_num = 0
            src_var = 0
            tgt_var = 0

            for (source_input, source_labels), (target_input, target_labels) in zip(self.source_loader, self.target_loader):
                # make sure the source and target has the same number of samples
                if source_input.size(0) > target_input.size(0):
                    source_input = source_input[:target_input.size(0)]
                    source_labels = source_labels[:target_input.size(0)]
                else:
                    target_input = target_input[:source_input.size(0)]
                    target_labels = target_labels[:source_input.size(0)]

                source_input = source_input.to(self.device)
                source_labels = source_labels.to(self.device)
                target_input = target_input.to(self.device)
                target_labels = target_labels.to(self.device)

                encoder_optimizer.zero_grad()

                # The architecture-specific forward pass 
                source_features = self.encoder(source_input, self.mask_dict)
                target_features = self.encoder(target_input, self.mask_dict)
                    
                source_outputs = self.source_classifier(source_features)
                target_outputs = self.target_classifier(target_features)

                source_loss = criterion(source_outputs, source_labels)
                target_loss = criterion(target_outputs, target_labels)

                loss = source_loss 

                # a value consider the inner class variance of the features
                cond_source_features = 0
                for i in range(10):
                    cond_source_features += torch.sum(torch.var(source_features[source_labels == i], dim=0))
                
                cond_target_variance = 0
                for i in range(10):
                    cond_target_variance += torch.sum(torch.var(target_features[target_labels == i], dim=0))

                # SFDA loss 
                if self.args.prune_method == 'admm-lda':
                    loss -= alpha*torch.clamp(target_loss, max=10) 
                    loss += sfda_multiclass_regularization_loss(target_features, target_labels, 10, 'target', lambda_reg=1e3)
                elif self.args.prune_method == 'admm-ntl':
                    loss -= alpha*torch.clamp(target_loss, max=1) * torch.clamp(MMD_loss()(source_features.view(source_features.size(0), -1), target_features.view(target_features.size(0), -1)), max=1) 
                
                # The admm loss is the loss + rho/2 * sum((param - Z + U)^2)
                # Compute ADMM regularization term with detached Z and U
                admm_reg = sum([torch.norm((param - Z_dict[name].detach() + U_dict[name].detach()))
                                for name, param in self.encoder.named_parameters() if param.requires_grad])

                # Compute the total ADMM loss
                admm_loss = loss + rho/2 * admm_reg
                admm_loss.backward()
                encoder_optimizer.step()
                
                
                # apply the mask to the model
                for name, param in self.encoder.named_parameters():
                    if name in self.mask_dict:
                        param.data = param.data * self.mask_dict[name]
                        # set the gradient to zero
                        param.grad = param.grad * self.mask_dict[name]

                # Record the admm loss
                loss_sum += loss.item()
                source_loss_sum += source_loss.item()
                target_loss_sum += target_loss.item()
                admm_loss_sum += admm_loss.item()
                sample_num += source_input.size(0)
                src_var += cond_source_features
                tgt_var += cond_target_variance

            # # how many percentage parameters are adjusted 
            # changed = 0
            # total = 0
            # for name, param in self.encoder.named_parameters():
            #     if param.requires_grad:
            #         changed += torch.sum(param.grad != 0).item()
            #         total += param.numel()
            # print(f"Percentage of changed parameters: {changed/total}")

            # Print the admm loss set in 2 demical values
            logging.info(f'Epoch {epoch}: admm loss: {admm_loss_sum / sample_num:.4f}; task loss: {loss_sum / sample_num:.4f}; source loss: {source_loss_sum / sample_num:.4f}; target loss: {target_loss_sum / sample_num:.4f}; source variance: {src_var/sample_num:.4f}; target variance: {tgt_var/sample_num:.4f};')
            all_admm_loss += admm_loss
            # logging.info(f"Epoch {epoch}: Source loss: {source_loss.item()}")
            # logging.info(f"Epoch {epoch}: Target loss: {target_loss.item()}")
        return all_admm_loss / nepoch
    

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
        self.encoder.train()
        for iteration in range(self.max_iterations):
            admm_loss = self.update_weights(Z_dict, U_dict, rho, alpha)
            # Update the Z variables
            l1_alpha = 1e-4
            Z_dict = self.update_Z_l1(U_dict, l1_alpha, rho)

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
            # Check whether there is an improvement in the loss to early stop 
            if iteration > 10:
                if abs(admm_loss - previous_loss) < 1e-4:
                    break
            previous_loss = admm_loss

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
    elif target_domain == 'imagenette':
        target_trainloader, target_testloader = get_imagenette_dataloader(args, ratio=finetune_ratio)
    elif target_domain == 'imagewoof':
        target_trainloader, target_testloader = get_imagewoof_dataloader(args, ratio=finetune_ratio)

    
    if args.arch == 'vgg11':
        encoder = VGGEncoder(model)
        classifier = VGGClassifier(model)
    elif args.arch == 'resnet18':
        encoder = ResNetEncoder(model)
        classifier = ResNetClassifier(model)

    # Initialize the ADMM pruner
    admm_pruner = ADMMEncoderPruner(encoder, classifier, source_trainloader, target_trainloader, args, max_iterations=5000, prune_percentage=0.99)
    admm_pruner.initialize_target_classifier()
    # Evaluate the model
    admm_pruner.evaluate(source_testloader)
    admm_pruner.evaluate(target_testloader)

    # Run the ADMM algorithm
    admm_pruner.run_admm()

    # Create the directory to save the model
    model_path = f'saved_models/{args.arch}/{args.prune_method}/{source_domain}_to_{target_domain}/{args.seed}/'
    model_dir = os.path.join(os.path.dirname(__file__), '../..', model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save the ADMM pruner as a pickle file
    # with open(f'saved_models/{args.arch}/{source_domain}_to_{target_domain}/admm_pruner.pkl', 'wb') as f:
    #     pickle.dump(admm_pruner, f)

    # Save the pruned model and masks
    torch.save(admm_pruner.encoder, model_path + '/admm_encoder.pth')
    torch.save(admm_pruner.source_classifier, model_path + '/admm_source_classifier.pth')
    torch.save(admm_pruner.mask_dict, model_path + '/admm_mask.pth')


if __name__ == '__main__':
    main()