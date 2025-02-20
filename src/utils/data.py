import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from .mnistm import MNISTM
from .syn import SyntheticDigits

from utils.args import get_args

def get_cifar_dataloader(args, ratio=1.0):
    """
    Get the CIFAR10 dataloader
    """
    # Data loading code for cifar10 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(len(train_dataset) * ratio)
    print(f"Using the sample size of {subset_size}.")

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = datasets.CIFAR10(
        root=args.data,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader

def get_usps_dataloader(args, ratio=1.0):
    """
    Get the USPS dataloader
    """
    # Data loading code for USPS 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    train_dataset = datasets.USPS(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(7291 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = datasets.USPS(
        root=args.data,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader

def get_mnist_dataloader(args, ratio=1.0):
    """
    Get the MNIST dataloader
    """
    # Data loading code for MNIST 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    train_dataset = datasets.MNIST(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(60000 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = datasets.MNIST(
        root=args.data,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader

def get_svhn_dataloader(args, ratio=1.0):
    """
    Get the SVHN dataloader
    """
    # Data loading code for SVHN 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970],
        ),
    ])

    train_dataset = datasets.SVHN(
        root=args.data,
        split='train',
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(73257 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = datasets.SVHN(
        root=args.data,
        split='test',
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader

def get_mnistm_dataloader(args, ratio=1.0):
    """
    Get the MNISTM dataloader
    """
    # Data loading code for MNISTM 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    train_dataset = MNISTM(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(60000 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = MNISTM(
        root=args.data,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader

def get_syn_dataloader(args, ratio=1.0):
    """
    Get the Synthetic Digits dataloader
    """
    # Data loading code for Synthetic Digits 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
    ])

    train_dataset = SyntheticDigits(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(60000 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(60000)
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = SyntheticDigits(
        root=args.data,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )

    return train_loader, val_loader

def get_cifar100_dataloader(args, ratio=1.0):
    """
    Get the CIFAR100 dataloader
    """
    # Data loading code for cifar100 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2673, 0.2564, 0.2762],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2673, 0.2564, 0.2762],
        ),
    ])

    train_dataset = datasets.CIFAR100(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(45000 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = datasets.CIFAR100(
        root=args.data,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader

def get_stl_dataloader(args, ratio=0.1):
    """
    Get the STL10 dataloader
    """
    # Data loading code for STL10 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
    ])

    train_dataset = datasets.STL10(
        root=args.data,
        split='train',
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(len(train_dataset)*ratio)
    print(f"Using sample size of {subset_size}.")
                      
    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = datasets.STL10(
        root=args.data,
        split='test',
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader

def get_imagenette_dataloader(args, ratio=0.1):
    # Define transformations

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create train and test dataset and dataloader
    data_path = args.data + "imagenette2/train"
    train_dataset = ImageFolder(root=data_path, transform=transform)

    # Define the size of the subset
    subset_size = int(len(train_dataset)*ratio)
    print(f"Using sample size of {subset_size}.")
                      
    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    data_path = args.data + "imagenette2/val"
    test_dataset = ImageFolder(root=data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader

def get_imagewoof_dataloader(args, ratio=0.1):
    # Define transformations

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create train and test dataset and dataloader
    data_path = args.data + "imagewoof2/train"
    train_dataset = ImageFolder(root=data_path, transform=transform)
    # Define the size of the subset
    subset_size = int(len(train_dataset)*ratio)
    print(f"Using sample size of {subset_size}.")
                      
    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    data_path = args.data + "imagewoof2/val"
    test_dataset = ImageFolder(root=data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader

if __name__ == "__main__":
    args = get_args()
    train_loader, val_loader = get_svhn_dataloader(args)
    # train_loader, val_loader = get_imagewoof_dataloader(args)
    # for i, (input, target) in enumerate(train_loader):
    #     print(input.shape, target.shape)
    #     break
    # for i, (input, target) in enumerate(val_loader):
    #     print(input.shape, target.shape)
    #     break
    # print("Done.")