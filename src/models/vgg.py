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