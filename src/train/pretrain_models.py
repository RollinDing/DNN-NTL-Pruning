import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

def pretrain_model():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and normalize CIFAR10
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to the input dimension of VGG
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization parameters for CIFAR10
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)

    # Modify VGG model
    vgg = models.vgg16(pretrained=True)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 10)  # Change the last layer for CIFAR10
    vgg.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = vgg(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += (outputs.argmax(dim=1) == labels).sum().item() / labels.size(0)
            # Display loss and accuracy every 10 mini-batches
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                print('[%d, %5d] accuracy: %.3f' %
                    (epoch + 1, i + 1, running_accuracy / 10))
                running_accuracy = 0.0
        
        # Validate the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = vgg(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item() / labels.size(0)

        print('Epoch %d: Accuracy of the network on the 10000 test images: %.3f %%' % (
            epoch + 1, 100 * correct / total))

    print('Finished Training')

if __name__ == '__main__':
    pretrain_model()