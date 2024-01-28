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
from utils.data import *

from models.vgg import PrunableVGG

import matplotlib.pyplot as plt
from tqdm import tqdm as tdqm
from copy import deepcopy

class ORGPruner:
    """
    The pruning will only be applied on the original dataset
    """
    def __init__(self, model, dataloader, prune_method='l1_unstructured', prune_percentage=0.1, source_perf_threshold=0.9, max_iterations=10):    
        self.dataloader = dataloader
        self.nepochs = 30
        self.lr = 0.001
        self.prune_method = prune_method
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

        self.importance_scores = self.compute_gradient_importance()
        
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
        print(f'Accuracy on source dataset: {correct / total}')
        return correct / total

    def fine_tune_model(self, dataloader):
        # Fine-tuning the model 
        self.model.train()

        # Only fine-tune the unfrozen parameters
        optimizer = torch.optim.SGD([param for name, param in self.model.named_parameters() if param.requires_grad], lr=self.lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.nepochs):
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs, self.mask_dict)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Reapply the masks to keep pruned weights at zero
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in self.mask_dict:
                            mask = self.mask_dict[name]
                            param.data.mul_(mask)

        # check the model sparsity by counting the number of non-zero parameters
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f'Layer {name} sparsity: {1 - torch.sum(param == 0).item() / param.numel()}')
        
        
    def compute_gradient_importance(self):
        """
        The importance score is not correct here --> we are going to find the important score to the source domain but the not for the target domain,
        The design of the score should be based on two SEPERATE loss function --> rather than one.
        """
        self.model.train()
        model_weights = [param for name, param in self.model.named_parameters() if param.requires_grad]
        gradients = self.compute_loader_gradients()

        # The importance scores are the gradients times the weights
        importance_scores = [torch.abs(gradients[name])*torch.abs(model_weights[i]) for i, name in enumerate(gradients)]
        # importance_scores = [torch.abs(gradients[name]) for i, name in enumerate(gradients)]

        # the importance score is the weights magnitude
        # importance_scores = [torch.abs(model_weights[i]) for i, name in enumerate(gradients)]
        return importance_scores

    def compute_loader_gradients(self):
        gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        for inputs, labels in self.dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs, self.mask_dict)
            loss = criterion(outputs, labels)

            # Backward pass
            self.model.zero_grad()  # Reset gradients to zero
            loss.backward()

            # Accumulate gradients
            for name, param in self.model.named_parameters():
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
        importance_scores = self.importance_scores

        # Flatten the importance scores and sort them
        all_scores = torch.cat([scores.flatten() for scores in importance_scores])
        threshold_idx = int(pruning_ratio * all_scores.numel())

        # Determine the pruning threshold
        threshold, _ = torch.kthvalue(all_scores, threshold_idx)

        # Apply the threshold to the importance scores and create a mask
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad and param.grad is not None:
                # Compute mask based on the importance score threshold
                mask = importance_scores[i] > threshold
                self.mask_dict[name] = mask.float()
                # print(f'Layer {i} sparsity: {1 - torch.sum(mask).item() / mask.numel()}')

                # Apply the mask to the weights and freeze pruned weights
                with torch.no_grad():
                    param.data.mul_(mask.float())

        # Return the mask dictionary, which now effectively freezes pruned weights
        # Evaluate the model after pruning
        self.evaluate(self.dataloader)
        # Evaluate the model after fine-tuning
        self.fine_tune_model(self.dataloader)
        self.evaluate(self.dataloader)
        return self.mask_dict
    
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

class NTLPruner:
    def __init__(self, model, source_loader, target_loader, prune_method='l1_unstructured', prune_percentage=0.1, source_perf_threshold=0.9, max_iterations=10):    
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.nepochs = 20
        self.lr = 1e-3
        self.prune_method = prune_method
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
        print(f'Accuracy on dataset: {correct / total}\n')
        return correct / total

    def fine_tune_model(self, dataloader, nepochs=30, lr=1e-3):
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

        # # check the model sparsity by counting the number of non-zero parameters
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f'Layer {name} sparsity: {1 - torch.sum(param == 0).item() / param.numel()}')

    def ntl_fine_tune_model(self, source_loader, target_loader, alpha=0.1, lr=1e-3):
        # Fine-tuning the model 
        self.model.train()

        # Only fine-tune the unfrozen parameters
        optimizer = torch.optim.SGD([param for name, param in self.model.named_parameters() if param.requires_grad], lr=lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.nepochs):
            for (source_inputs, source_labels), (target_inputs, target_labels) in zip(source_loader, target_loader):

                source_inputs = source_inputs.to(self.device)
                source_labels = source_labels.to(self.device)

                target_inputs = target_inputs.to(self.device)
                target_labels = target_labels.to(self.device)

                optimizer.zero_grad()
                
                source_outputs = self.model(source_inputs, self.mask_dict)
                source_loss = criterion(source_outputs, source_labels)

                target_outputs = self.model(target_inputs, self.mask_dict)
                target_loss = criterion(target_outputs, target_labels)
                
                differential_loss = source_loss + torch.log(1+alpha*source_loss/target_loss)
                
                differential_loss.backward()
                optimizer.step()

        # # check the model sparsity by counting the number of non-zero parameters
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f'Layer {name} sparsity: {1 - torch.sum(param == 0).item() / param.numel()}')

    def compute_gradient_importance(self):
        """
        The importance score is not correct here --> we are going to find the important score to the source domain but the not for the target domain,
        The design of the score should be based on two SEPERATE loss function --> rather than one.
        """
        self.model.train()
        model_weights = [param for name, param in self.model.named_parameters() if param.requires_grad]
        source_gradients = self.compute_loader_gradients(self.source_loader)
        target_gradients = self.compute_loader_gradients(self.target_loader)

        # Combine the gradients from both loaders to compute importance
        # For example, importance could be higher when source gradient is high and target gradient is low
        impact_source = [source_gradients[name] for i, name in enumerate(source_gradients)]
        impact_target = [target_gradients[name] for i, name in enumerate(target_gradients)]

        # # standardize two importance scores
        # impact_source_std = [(scores-torch.min(scores))/(torch.max(scores)-torch.min(scores)) for scores in impact_source]
        # impact_target_std = [(scores-torch.min(scores))/(torch.max(scores)-torch.min(scores)) for scores in impact_target]

        # min_impact_source = torch.min(torch.cat([scores.flatten() for scores in impact_source]))
        # max_impact_source = torch.max(torch.cat([scores.flatten() for scores in impact_source]))
        # min_impact_target = torch.min(torch.cat([scores.flatten() for scores in impact_target]))
        # max_impact_target = torch.max(torch.cat([scores.flatten() for scores in impact_target]))

        # impact_source_std = [(scores-min_impact_source)/(max_impact_source-min_impact_source) for scores in impact_source]
        # impact_target_std = [(scores-min_impact_target)/(max_impact_target-min_impact_target) for scores in impact_target]

        # TODO: how to design the importance score?
        # importance_scores = [importance_scores_source[i]*torch.log(1+importance_scores_source_std[i]/(importance_scores_target_std[i]+1e-6)) for i in range(len(importance_scores_source))]
        # importance_scores = [impact_source_std[i]/(impact_target_std[i]+1e-12) for i in range(len(impact_target))]
        # importance_scores = [importance_scores_source[i] + torch.log(1+importance_scores_source[i]/importance_scores_target[i]) for i in range(len(importance_scores_source))]
        

        # The importance score is defined as the importance score on the target dataset
        # epsilon = 1e-6

        # print the number of impact_source which smaller than epsilon
        # print(torch.sum(torch.cat([scores.flatten() for scores in impact_source]) < epsilon)/len(torch.cat([scores.flatten() for scores in impact_source])))

        # importance_scores = [torch.where(impact_source[i] < epsilon, 
        #                                  impact_target[i], 
        #                                  1e-15*torch.ones_like(model_weights[i])) for i in range(len(impact_source))]

        alpha=1
        importance_scores = [impact_source[i]-impact_target[i] for i in range(len(impact_target))]
        # importance_scores = impact_source - impact_target
        # minmax normalization
        importance_scores = [(scores-torch.min(scores))/(torch.max(scores)-torch.min(scores)) for scores in importance_scores]
        

        return importance_scores
    
    def compute_loader_gradients(self, dataloader):
        # Consider using the NTL loss for finding the importance gradient
        gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs, self.mask_dict)
            loss = criterion(outputs, labels)

            # Backward pass
            self.model.zero_grad()  # Reset gradients to zero
            loss.backward()

            # Accumulate gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Due to cross-entropy loss, the last layer will not have gradients
                    # The absolute value of the gradients and the ones that are in the masks should be zero out
                    gradients[name] += (torch.abs(param.grad)+1e-12)*self.mask_dict[name]

        return gradients

    def compute_ntl_gradients_importance(self):
        """
        The importance score is not correct here --> we are going to find the important score to the source domain but the not for the target domain,
        The design of the score should be based on two SEPERATE loss function --> rather than one.
        """
        self.model.train()
        model_weights = [param for name, param in self.model.named_parameters() if param.requires_grad]
        ntl_gradients = self.compute_ntl_gradients()

        # The importance scores are the gradients times the weights
        importance_scores = [ntl_gradients[name]*torch.abs(model_weights[i]) for i, name in enumerate(ntl_gradients)]

        # minimax normalization
        importance_scores = [(scores-torch.min(scores))/(torch.max(scores)-torch.min(scores)) for scores in importance_scores]
        return importance_scores  
    
    def compute_ntl_gradients(self, alpha=1e3):
        # Consider using the NTL loss for finding the importance gradient
        gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        for (source_inputs, source_labels), (target_inputs, target_labels) in zip(self.source_loader, self.target_loader):
            source_inputs = source_inputs.to(self.device)
            source_labels = source_labels.to(self.device)
            source_outputs = self.model(source_inputs, self.mask_dict)
            source_loss = criterion(source_outputs, source_labels)

            target_inputs = target_inputs.to(self.device)
            target_labels = target_labels.to(self.device)
            target_outputs = self.model(target_inputs, self.mask_dict)
            target_loss = criterion(target_outputs, target_labels)

            total_loss = source_loss-alpha*target_loss
            # total_loss = source_loss + torch.log(1+alpha*source_loss/target_loss)
            # Backward pass
            self.model.zero_grad()  # Reset gradients to zero
            total_loss.backward()

            # Accumulate gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Due to cross-entropy loss, the last layer will not have gradients
                    # The absolute value of the gradients and the ones that are in the masks should be zero out
                    gradients[name] += (torch.abs(param.grad)+1e-12)*self.mask_dict[name]

        return gradients

    def prune(self, pruning_ratio):
        """
        Prune the model based on the calculated importance scores.

        Args:
        pruning_ratio (float): The ratio of weights to prune (0.0 to 1.0).
        """
        print('Pruning the model with pruning ratio:', pruning_ratio)
        # Calculate importance scores
        # importance_scores = self.compute_gradient_importance()
        importance_scores = self.compute_ntl_gradients_importance()

        # Flatten the importance scores and sort them
        all_scores = torch.cat([scores.flatten() for scores in importance_scores])
        threshold_idx = int(pruning_ratio * all_scores.numel())

        # Determine the pruning threshold
        threshold, _ = torch.kthvalue(all_scores, threshold_idx)
        print(f'\nThreshold: {threshold} \n')

        # Apply the threshold to the importance scores and create a mask
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad and param.grad is not None:
                # Compute mask based on the importance score threshold
                mask = importance_scores[i] > threshold
                self.mask_dict[name] = mask.float()
                print(f'Layer {i} sparsity: {1 - torch.sum(mask).item() / mask.numel()}')

                # Apply the mask to the weights and freeze pruned weights
                with torch.no_grad():
                    param.data.mul_(mask.float())

        # Return the mask dictionary, which now effectively freezes pruned weights
                    
        # TODO: How to finetune the pruned model? 
    
        # Evaluate the model after pruning
        print("\n")
        print("Before fine-tuning")
        print(f'Evaluate on source loader')
        self.evaluate(self.source_loader)
        print(f'Evaluate on target loader')
        self.evaluate(self.target_loader)

        # Evaluate the model after fine-tuning
        self.fine_tune_model(self.source_loader)
        print("After fine-tuning")
        print(f'Evaluate on source loader')
        self.evaluate(self.source_loader)
        print(f'Evaluate on target loader')
        self.evaluate(self.target_loader)
        return self.mask_dict
    
    def iterative_prune(self, pruning_ratio, iterations=10):
        """
        Prune the model based on the calculated importance scores.

        Args:
        pruning_ratio (float): The ratio of weights to prune (0.0 to 1.0).
        """
        print('Pruning the model with pruning ratio:', pruning_ratio)
        
        for iteration in range(iterations):
            print(f'\nIteration {iteration + 1}/{iterations}')

            # Calculate importance scores
            # importance_scores = self.compute_gradient_importance()
            importance_scores = self.compute_ntl_gradients_importance()
            
            # Flatten the importance scores and select those are not pruned
            all_scores = torch.cat([scores.flatten() for scores in importance_scores])
            # To verify count the number of 0 in all scores
            print("Number of zero in the importance score", torch.sum(all_scores==0)/len(all_scores))

            all_scores = all_scores[all_scores!=0]
            all_scores = all_scores[~torch.isnan(all_scores)]
            # all_scores = all_scores[all_scores < 1]
            threshold_idx = int(pruning_ratio*all_scores.numel())

            # Determine the pruning threshold
            threshold, _ = torch.kthvalue(all_scores, threshold_idx)
            print(f'\nThreshold: {threshold}')
            # if threshold >=1:
            #     print("Reaching the limit!!\n")
            #     break
            # Apply the threshold to the importance scores and create a mask
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if param.requires_grad and param.grad is not None:
                    # Compute mask based on the importance score threshold
                    mask = importance_scores[i] > threshold
                    # mask += importance_scores[i] == 1
                    self.mask_dict[name] *= mask.float()
                    # print(f'Layer {i} sparsity: {1 - torch.sum(self.mask_dict[name]).item() / mask.numel()}')
                    # Apply the mask to the weights and freeze pruned weights
                    with torch.no_grad():
                        param.data.mul_(mask.float())
        
            # Return the mask dictionary, which now effectively freezes pruned weights
        
            # Evaluate the model after pruning
            print("\n")
            print("Before fine-tuning")
            print("Model Sparsity: {0}".format(self.model_sparsity()))
            print(f'Evaluate on source loader')
            self.evaluate(self.source_loader)
            print(f'Evaluate on target loader')
            self.evaluate(self.target_loader)

            # Evaluate the model after fine-tuning
            # combine the source and target dataset to fine-tune the model at the ratio of 10:1
            target_ratio = 0.05
            source_dataset = self.source_loader.dataset
            target_dataset = self.target_loader.dataset

            subset_size = int(len(source_dataset) * target_ratio)
            indices = np.random.permutation(len(source_dataset))
            source_indices = indices[:subset_size]
            source_subset = Subset(source_dataset, source_indices)

            subset_size = int(len(target_dataset) * target_ratio)
            indices = np.random.permutation(len(target_dataset))
            target_indices = indices[:subset_size]
            target_subset = Subset(target_dataset, target_indices)

            finetune_dataset = torch.utils.data.ConcatDataset([source_subset, target_subset])
            finetune_dataloader = torch.utils.data.DataLoader(
                finetune_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=4,
            )

            self.ntl_fine_tune_model(self.source_loader, self.target_loader, alpha=1e4, lr=1e-4)

            print("After NTL fine-tuning")
            print(f'Evaluate on source loader')
            self.evaluate(self.source_loader)
            print(f'Evaluate on target loader')
            self.evaluate(self.target_loader)

            # self.fine_tune_model(finetune_dataloader, lr=1e-4)
            
            # print("After fine-tuning")
            # print(f'Evaluate on source loader')
            # self.evaluate(self.source_loader)
            # print(f'Evaluate on target loader')
            # self.evaluate(self.target_loader)

        return self.mask_dict        
            
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
    
    def restore_model(self, iteration):
        # Logic to restore the model to a previous state
        pass

    def selective_prune(self):
        original_performance_source = self.evaluate(self.source_loader)
        original_performance_target = self.evaluate(self.target_loader)

        for iteration in range(self.max_iterations):
            mask = self.prune(0.1)
            # TODO: the fine-tune here can be NTL?
            self.fine_tune_model()

            current_performance_source = self.evaluate(self.source_loader)
            current_performance_target = self.evaluate(self.target_loader)

            if current_performance_source < self.source_perf_threshold:
                self.restore_model(iteration)
                break

            if current_performance_target < original_performance_target:
                original_performance_target = current_performance_target

        return self.model

def load_base_model(model, model_name, source_domain, source_trainloader, source_testloader):
    base_model_dir = f'base_models/{model_name}-{source_domain}.pth'
    if os.path.exists(base_model_dir):
        print('Loading the base model from {}'.format(base_model_dir))
        model.load_state_dict(torch.load(base_model_dir))
    else:
        # finetune the model on source dataset 
        print('Finetune the model on source dataset')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nepochs = 100
        model.to(device)    
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        # adjust the learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        best_acc = 0
        patience = 10
        for epoch in range(nepochs):
            model.train()
            for inputs, labels in tdqm(source_trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step() 
            scheduler.step()

            # validate the model
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for input, labels in source_testloader:
                    input = input.to(device)
                    labels = labels.to(device)
                    outputs = model(input)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Epoch {epoch+1}/{nepochs}, Accuracy on source dataset: {correct / total}')

            # Early stopping
            if correct / total > best_acc:
                best_acc = correct / total
                # save the best model
                print("Saving the best model")
                torch.save(model.state_dict(), base_model_dir)
                patience=10
            else:
                patience-=1
                if patience == 0:
                    print('Early stopping at epoch:', epoch+1)
                    break

        print("Finish fine-tune the model, the best accuracy is:", best_acc)
    return model
    
def evaluate_original_transferability(model2prune, source_trainloader, source_testloader, target_trainloader, target_testloader):
    pruner = ORGPruner(model2prune, source_trainloader)
    # model sparsity before pruning
    print('Model sparsity:', pruner.model_sparsity())
    pruner.evaluate(source_testloader)

    # prune the model with 10% sparsity
    mask_dict = pruner.prune(0.98)
    # model sparsity after pruning
    print('Model sparsity:', pruner.model_sparsity())
    print("After pruning on the source dataset")
    pruner.evaluate(source_testloader)

    # finetune and evaluate the model on target dataset
    pruner.fine_tune_model(target_trainloader)
    print('Model sparsity:', pruner.model_sparsity())
    print("After fine-tuning on the target dataset")
    pruner.evaluate(target_testloader)

if __name__ == '__main__':
    # Example usage 
    # Set the random seed for reproducible experiments
    torch.manual_seed(1234)
    np.random.seed(1234)
    num_classes = 10
    # import the pretrained torchvision model 
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
    # load args 
    args = get_args()
    
    source_domain = 'mnist'
    target_domain = 'usps'
    finetune_ratio = 0.1

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

    model = load_base_model(model, 'vgg11', source_domain, source_trainloader, source_testloader)

    # prune the model
    for param in model.parameters():
        param.requires_grad = True
    # build the prunable model
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
    

    # traditional pruning with a fixed pruning ratio
    modelcopy = deepcopy(model2prune)
    
    evaluate_original_transferability(modelcopy, source_trainloader, source_testloader, target_trainloader, target_testloader)

    # Iterative NTL pruning with a fixed pruning ratio 
    pruner = NTLPruner(model2prune, source_trainloader, target_trainloader)
    # model sparsity before pruning
    print('Model sparsity:', pruner.model_sparsity())
    pruner.evaluate(source_testloader)

    # prune the model with 20% sparsity
    mask_dict = pruner.iterative_prune(0.22, iterations=10)
    pruner.ntl_fine_tune_model(source_trainloader, target_trainloader, alpha=100, lr=1e-4)
    print("After NTL fine-tuning on the target dataset")

    # save the model
    model_dir = os.path.join(os.path.dirname(__file__), '../..', f'saved_models/{source_domain}_to_{target_domain}')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    torch.save(pruner.model, f'{model_dir}/itp_model.pth')
    torch.save(mask_dict, f'{model_dir}/itp_mask.pth')

    # model sparsity after pruning
    print('Model sparsity:', pruner.model_sparsity())
    pruner.evaluate(source_testloader)

    # finetune and evaluate the model on target dataset
    print("Before fine-tuning on the target dataset")
    print('Model sparsity:', pruner.model_sparsity())
    pruner.evaluate(target_testloader)
    pruner.fine_tune_model(target_trainloader, lr=1e-4)

    print("After fine-tuning on the target dataset")
    print('Model sparsity:', pruner.model_sparsity())
    pruner.evaluate(target_testloader)

    exit()
    pruner = Pruner(model, mnist_trainloader, usps_trainloader)
    pruner.fine_tune_model()
    print('Model sparsity:', pruner.model_sparsity())
    print(pruner.evaluate(mnist_testloader))
    print(pruner.evaluate(usps_testloader))
    importance_scores = pruner.compute_gradient_importance()
    mask_dict = pruner.prune(1e-4)
    pruner.fine_tune_model()
    print('Model sparsity:', pruner.model_sparsity())
    print(pruner.evaluate(mnist_testloader))
    print(pruner.evaluate(usps_testloader))
    # visualzie the importance scores with histogram
    layer = 2
    fig, ax = plt.subplots()
    data = importance_scores[layer].cpu().detach().numpy().flatten()
    ax.set_title(f'importance scores for layer {layer}')
    ax.hist(data)
    # set y scale 
    ax.set_yscale('log')
    plt.savefig(f'imgs/importance_scores[{layer}].png')
    plt.show()


