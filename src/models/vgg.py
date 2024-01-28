import torch
import torch.nn as nn
import torchvision.models as models

class PrunableVGG(nn.Module):
    def __init__(self, original_vgg):
        super(PrunableVGG, self).__init__()
        self.features = original_vgg.features
        self.avgpool = original_vgg.avgpool
        self.classifier = original_vgg.classifier

    def forward(self, x, weight_mask):
        # Apply masks to the features
        for name, module in self.features.named_children():
            if isinstance(module, nn.Conv2d):
                if name in weight_mask:
                    original_weight = module.weight.data.clone()
                    module.weight.data *= weight_mask[name]
                else:
                    original_weight = None

                x = module(x)

                if original_weight is not None:
                    module.weight.data = original_weight
            else:
                x = module(x)

        # Apply avgpool and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Apply masks to the classifier
        for name, module in self.classifier.named_children():
            if isinstance(module, nn.Linear):
                if name in weight_mask:
                    original_weight = module.weight.data.clone()
                    module.weight.data *= weight_mask[name]
                else:
                    original_weight = None

                x = module(x)

                if original_weight is not None:
                    module.weight.data = original_weight
            else:
                x = module(x)

        return x
    
class PrunableResNet18(nn.Module):
    def __init__(self, original_resnet):
        super(PrunableResNet18, self).__init__()
        self.conv1 = original_resnet.conv1
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4
        self.avgpool = original_resnet.avgpool
        self.fc = original_resnet.fc

    def forward(self, x, weight_mask):
        # Apply masks to the features during resnet18 forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for name, module in self.layer1.named_children():
            if isinstance(module, nn.Conv2d):
                if name in weight_mask:
                    original_weight = module.weight.data.clone()
                    module.weight.data *= weight_mask[name]
                else:
                    original_weight = None

                x = module(x)

                if original_weight is not None:
                    module.weight.data = original_weight
            else:
                x = module(x)

        for name, module in self.layer2.named_children():
            if isinstance(module, nn.Conv2d):
                if name in weight_mask:
                    original_weight = module.weight.data.clone()
                    module.weight.data *= weight_mask[name]
                else:
                    original_weight = None

                x = module(x)

                if original_weight is not None:
                    module.weight.data = original_weight
            else:
                x = module(x)

        for name, module in self.layer3.named_children():
            if isinstance(module, nn.Conv2d):
                if name in weight_mask:
                    original_weight = module.weight.data.clone()
                    module.weight.data *= weight_mask[name]
                else:
                    original_weight = None

                x = module(x)

                if original_weight is not None:
                    module.weight.data = original_weight
            else:
                x = module(x)

        for name, module in self.layer4.named_children():
            if isinstance(module, nn.Conv2d):
                if name in weight_mask:
                    original_weight = module.weight.data.clone()
                    module.weight.data *= weight_mask[name]
                else:
                    original_weight = None

                x = module(x)

                if original_weight is not None:
                    module.weight.data = original_weight
            else:
                x = module(x)

        # Apply avgpool and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Apply masks to the classifier
        for name, module in self.fc.named_children():
            if isinstance(module, nn.Linear):
                if name in weight_mask:
                    original_weight = module.weight.data.clone()
                    module.weight.data *= weight_mask[name]
                else:
                    original_weight = None

                x = module(x)

                if original_weight is not None:
                    module.weight.data = original_weight
            else:
                x = module(x)

        return x
    


