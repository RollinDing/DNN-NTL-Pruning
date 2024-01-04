import torch
import torch.nn.utils.prune as prune
import torchvision 

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../prune')))

from args import get_args
from data import get_usps_dataloader, get_mnist_dataloader

import matplotlib.pyplot as plt
from tqdm import tqdm as tdqm


class ORGPruner:
    """
    The pruning will only be applied on the original dataset
    """
    def __init__(self, model, dataloader, prune_method='l1_unstructured', prune_percentage=0.1, source_perf_threshold=0.9, max_iterations=10):    
        self.dataloader = dataloader
        self.nepochs = 20
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
                # apply the mask to the model
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data.mul_(self.mask_dict[name])
                outputs = self.model(input)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy on source dataset: {correct / total}')
        return correct / total

    def fine_tune_model(self):
        # Fine-tuning the model 
        self.model.train()

        # Only fine-tune the unfrozen parameters
        optimizer = torch.optim.SGD([param for name, param in self.model.named_parameters() if param.requires_grad], lr=self.lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.nepochs):
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
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
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f'Layer {name} sparsity: {1 - torch.sum(param == 0).item() / param.numel()}')
        
        
    def compute_gradient_importance(self):
        """
        The importance score is not correct here --> we are going to find the important score to the source domain but the not for the target domain,
        The design of the score should be based on two SEPERATE loss function --> rather than one.
        """
        self.model.train()
        model_weights = [param for name, param in self.model.named_parameters() if param.requires_grad]
        gradients = self.compute_loader_gradients()

        # The importance scores are the gradients times the weights
        # importance_scores = [torch.abs(gradients[name])*model_weights[i] for i, name in enumerate(gradients)]
        importance_scores = [torch.abs(gradients[name]) for i, name in enumerate(gradients)]

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
            outputs = self.model(inputs)
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
                print(f'Layer {i} sparsity: {1 - torch.sum(mask).item() / mask.numel()}')

                # Apply the mask to the weights and freeze pruned weights
                with torch.no_grad():
                    param.data.mul_(mask.float())

        # Return the mask dictionary, which now effectively freezes pruned weights
        # Evaluate the model after pruning
        self.evaluate(self.dataloader)
        # Evaluate the model after fine-tuning
        self.fine_tune_model()
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
        self.lr = 0.001
        self.prune_method = prune_method
        self.prune_percentage = prune_percentage
        self.source_perf_threshold = source_perf_threshold
        self.max_iterations = max_iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.importance_scores = self.compute_gradient_importance()
        
    def evaluate(self, data_loader):
        # Evaluation the model 
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for input, labels in data_loader:
                input = input.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def fine_tune_model(self):
        # Fine-tuning the model 
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # Check the percentage of model parameters that are frozen
        num_frozen = 0
        num_total = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                num_total += param.numel()
                if not param.requires_grad:
                    num_frozen += param.numel()
        print(f'Percentage of frozen parameters: {num_frozen / num_total}')

        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(self.nepochs):
            for inputs, labels in self.source_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        pass

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
        importance_scores = [torch.abs(source_gradients[name] - target_gradients[name])*model_weights[i] for i, name in enumerate(source_gradients)]
        return importance_scores

    def compute_loader_gradients(self, loader):
        total_gradients = {}
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
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
        # Calculate importance scores
        importance_scores = self.importance_scores

        # Flatten the importance scores and sort them
        all_scores = torch.cat([scores.flatten() for scores in importance_scores])
        threshold_idx = int(pruning_ratio * all_scores.numel())
        
        # Determine the pruning threshold
        threshold, _ = torch.kthvalue(all_scores, threshold_idx)
        mask_dict = {}

        # Apply the threshold to the importance scores and create a mask
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                # Compute mask based on the importance score threshold
                mask = importance_scores[i] > threshold
                mask_dict[name] = mask.float()

                # Apply the mask to the weights and freeze pruned weights
                with torch.no_grad():
                    param.data.mul_(mask.float())

                    # Set requires_grad to False for pruned weights
                    if torch.any(mask == 0):
                        param.requires_grad = False

        # Return the mask dictionary, which now effectively freezes pruned weights
        return mask_dict
    
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



if __name__ == '__main__':
    # Example usage 
    # import the pretrained torchvision model 
    model = torchvision.models.vgg11(pretrained=True)
    # change the output layer to 10 classes (for digits dataset)
    model.fc = torch.nn.Linear(512, 10)

    # load args 
    args = get_args()

    # Load the source dataset
    mnist_trainloader, mnist_testloader = get_mnist_dataloader(args, ratio=0.1)

    # finetune the model on source dataset 
    print('Finetune the model on source dataset')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nepochs = 20
    model.to(device)    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(nepochs):
        for inputs, labels in tdqm(mnist_trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
        print(f'Epoch {epoch} loss: {loss.item()}')
    
    # evaluate the model on source dataset
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for input, labels in mnist_testloader:
            input = input.to(device)
            labels = labels.to(device)
            outputs = model(input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on source dataset: {correct / total}')

    # prune the model
    for param in model.parameters():
        param.requires_grad = True
    pruner = ORGPruner(model, mnist_trainloader)
    # model sparsity before pruning
    print('Model sparsity:', pruner.model_sparsity())
    # prune the model with 10% sparsity
    mask_dict = pruner.prune(0.9)
    # model sparsity after pruning
    print('Model sparsity:', pruner.model_sparsity())
    exit()

    usps_trainloader, usps_testloader = get_usps_dataloader(args, ratio=0.2)

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


