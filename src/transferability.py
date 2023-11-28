import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './utils')))

from args import get_args
from load_model import load_pretrained_model, load_pruned_model
from data import *
from pruning_utils import *
from prune_model import *

import logging
import time

def evaluate_model_transferability(args, logger, student='mnist', prun_iter=0):
    """
    Evaluate the model transferability
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = load_pruned_model(args, logger, prun_iter=prun_iter)

    ratio = 0.1
    if student == 'mnist':
        # Get the MNIST dataloader
        logger.info("=> loading MNIST dataset")
        train_loader, val_loader = get_mnist_dataloader(args, ratio=ratio)
        num_classes = 10
    elif student == 'cifar100':
        # Get the CIFAR100 dataloader
        logger.info("=> loading CIFAR100 dataset")
        train_loader, val_loader = get_cifar100_dataloader(args, ratio=ratio)
        num_classes = 100
    
    # transfer the pretrained model to the student dataset 
    student_model = copy.deepcopy(teacher_model)
    # modify the output layer for the student model, the student model is DataParallel object
    if args.arch.startswith('resnet'):
        student_model.module.fc = nn.Linear(student_model.module.fc.in_features, num_classes)
    elif args.arch.startswith('vgg'):
        student_model.module.classifier[6] = nn.Linear(student_model.module.classifier[6].in_features, num_classes)
    student_model.to(device)

    # finetune the student model on student dataset 
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    for epoch in range(30):  # loop over the dataset multiple times
        print('epoch: {}'.format(epoch))
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = student_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += (outputs.argmax(dim=1) == labels).sum().item() / labels.size(0)
            # Display loss and accuracy every 10 mini-batches
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                print('[%d, %5d] accuracy: %.3f' %
                    (epoch + 1, i + 1, running_accuracy / 10))

        scheduler.step()
    
    # evaluate the student model's accuracy
    student_accuracy = validate(val_loader, student_model, criterion, args)
    logger.info("=> student model accuracy: {}".format(student_accuracy))



if __name__ == '__main__':
    args = get_args()

    # Set up logging and logging directory and logging file using the current time
    logging.basicConfig(level=logging.INFO, format='%(message)s', 
                        handlers=[logging.FileHandler(os.path.join('logs/', time.strftime("%Y%m%d-%H%M%S") + '.log')), 
                        logging.StreamHandler()])
    
    logger = logging.getLogger(__name__)
    logger.info("Test the pruned model's transferability")
    logger.info(args)

    evaluate_model_transferability(args, logger, student='mnist', prun_iter=18)

    
    

