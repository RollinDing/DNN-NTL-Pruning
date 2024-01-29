"""
Implement Encoders and Classifications here
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

class ResNetEncoder(nn.Module):
    def __init__(self, original_model):
        super(ResNetEncoder, self).__init__()
        # Copy layers from the original model up to the average pooling layer
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def apply_mask(self, module, mask):
        original_weight = module.weight.data.clone()
        module.weight.data *= mask
        if module.bias is not None:
            original_bias = module.bias.data.clone()
            module.bias.data *= mask
        else:
            original_bias = None
        return original_weight, original_bias

    def restore_weight(self, module, original_weight, original_bias):
        module.weight.data = original_weight
        if original_bias is not None:
            module.bias.data = original_bias

    def forward(self, x, weight_masks):
        for feature in self.features:
            if isinstance(feature, nn.Sequential):
                for name, module in feature.named_children():
                    if isinstance(module, nn.Conv2d) and name in weight_masks:
                        original_weight, original_bias = self.apply_mask(module, weight_masks[name])
                        x = module(x)
                        self.restore_weight(module, original_weight, original_bias)
                    else:
                        x = module(x)
            else:
                x = feature(x)
        return x

class ResNetClassifier(nn.Module):
    def __init__(self, original_model, num_classes=1000):
        super(ResNetClassifier, self).__init__()
        self.num_classes = num_classes
        # The average pooling layer and fully connected layer
        self.avgpool = original_model.avgpool
        self.classifier = original_model.fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10('data', train=True, download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=10, drop_last=False, shuffle=True)

    test_dataset = datasets.CIFAR10('data/', train=False, download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=False)
    return train_loader, test_loader

    
if __name__ == "__main__":
    # Load and convert the pre-trained ResNet-18 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    original_model = models.resnet18(pretrained=True).to(device)
    checkpoint = torch.load('base_models/resnet18-simclr-cifar10.tar', map_location=device)
    resnet_encoder = ResNetEncoder(original_model)
    resnet_classifier = ResNetClassifier(original_model)

    optimizer = torch.optim.Adam(resnet_classifier.parameters(), lr=0.0003, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_cifar10_data_loaders(download=True)

    mask_dict = {}
    for name, module in resnet_encoder.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask_dict[name] = torch.ones(module.weight.shape).to(device)

    # Forward pass through the encoder and classifier with mask
    epochs = 100
    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            features = resnet_encoder(x_batch, mask_dict)
            logits = resnet_classifier(features)
            loss = criterion(logits, y_batch)
            
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            features = resnet_encoder(x_batch, mask_dict)
            logits = resnet_classifier(features)
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 Train accuracy: {top5_accuracy.item()}\t")