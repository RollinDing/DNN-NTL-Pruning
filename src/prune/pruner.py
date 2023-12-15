import torch
import torch.nn.utils.prune as prune

# import utils's path 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../prune')))

from args import get_args
from data import get_usps_dataloader, get_mnist_dataloader

import matplotlib.pyplot as plt

class Pruner:
    def __init__(self, model, source_loader, target_loader, prune_method='l1_unstructured', prune_percentage=0.1, source_perf_threshold=0.9, max_iterations=10):
        
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.prune_method = prune_method
        self.prune_percentage = prune_percentage
        self.source_perf_threshold = source_perf_threshold
        self.max_iterations = max_iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.importance_scores = self.compute_gradient_importance()
        
    def evaluate(self, data_loader):
        # Evaluation logic here
        pass

    def fine_tune_model(self):
        # Fine-tuning logic here
        pass

    def compute_gradient_importance(self):
        self.model.train()
        model_weights = [param for name, param in self.model.named_parameters() if param.requires_grad]
        source_gradients = self.compute_loader_gradients(self.source_loader)
        target_gradients = self.compute_loader_gradients(self.target_loader)

        # Combine the gradients from both loaders to compute importance
        # For example, importance could be higher when source gradient is high and target gradient is low
        importance_scores = [torch.abs(source_gradients[name] - target_gradients[name])*model_weights[i] for i, name in enumerate(source_gradients)]
        return importance_scores

    def compute_loader_gradients(self, loader):
        total_gradients = {}
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Aggregate gradients for each parameter
                    if name not in total_gradients:
                        total_gradients[name] = param.grad.clone()
                    else:
                        total_gradients[name] += param.grad.clone()

        # Average the gradients over the number of batches
        for name in total_gradients:
            total_gradients[name] /= len(loader)

        return total_gradients
    
    def prune(self, pruning_ratio):
        """
        Prune the model based on the calculated importance scores.

        Args:
        pruning_ratio (float): The ratio of weights to prune (0.0 to 1.0).
        """
        # Flatten the importance scores and sort them
        all_scores = torch.cat([scores.flatten() for scores in self.importance_scores])
        threshold_idx = int(pruning_ratio * all_scores.numel())
        
        # Determine the pruning threshold
        threshold, _ = torch.kthvalue(all_scores, threshold_idx)
        mask_dict = {}

        # Apply the threshold to the importance scores and create a mask
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                # Compute mask based on the importance score threshold
                mask = self.importance_scores[i] > threshold
                mask_dict[name] = mask.float()

                # Apply the mask to the weights
                param.data.mul_(mask.float())

                # Optionally, zero out the gradients of the pruned weights
                if param.grad is not None:
                    param.grad.data.mul_(mask.float())

        # The mask_dict contains the masks for each layer, which could be useful for analysis or debugging
        return mask_dict
    
    def restore_model(self, iteration):
        # Logic to restore the model to a previous state
        pass

    def selective_prune(self):
        original_performance_source = self.evaluate(self.source_loader)
        original_performance_target = self.evaluate(self.target_loader)

        for iteration in range(self.max_iterations):
            mask = self.prune()
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


if __name__ == '__main__':
    # Example usage 

    # a model for mnist and usps 
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.fc = torch.nn.Linear(512, 10)

    args = get_args()
    mnist_trainloader, minst_testloader = get_mnist_dataloader(args, ratio=0.1)
    usps_trainloader, usps_testloader = get_usps_dataloader(args, ratio=0.1)

    pruner = Pruner(model, mnist_trainloader, usps_trainloader)
    importance_scores = pruner.compute_gradient_importance()
    mask_dict = pruner.prune(0.1)
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


