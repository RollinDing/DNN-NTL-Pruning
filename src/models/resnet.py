# Load the pretrained model of ResNet50 which used the SIMCLR
from transformers import AutoModel
import torch

def get_resnet50_simclr(pretrained=True):
    # Load the model
    model = AutoModel.from_pretrained("edadaltocg/resnet50_simclr_cifar10")

    # If you need to use the model for inference
    model.eval()
    return model

if __name__ == "__main__":
    model = get_resnet50_simclr()
    # TODO: The model's weight (normalization layers) are not frozen
    # TODO: Think about a way to deal with the normalization layers
    print(model)
    print(model(torch.randn(1, 3, 32, 32)))

