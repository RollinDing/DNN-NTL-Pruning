"""
Implement Encoders and Classifications here
"""
import torch
import torch.nn as nn
import torchvision.models as models

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


if __name__ == "__main__":
    # Load and convert the pre-trained ResNet-18 model
    original_model = models.resnet50(pretrained=True)
    resnet_encoder = ResNetEncoder(original_model)
    resnet_classifier = ResNetClassifier(original_model)

    # Example input (random) and mask
    input_tensor = torch.rand(1, 3, 224, 224)
    weight_mask = {'0': torch.rand_like(original_model.layer1[0].conv1.weight)}

    # Forward pass through the encoder and classifier with mask
    features = resnet_encoder(input_tensor, weight_mask)
    output = resnet_classifier(features)

    print(output.shape) 