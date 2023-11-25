import torch
import torch.nn as nn
import torchvision.models as models

import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './prune')))

from args import get_args
from data import get_cifar_dataloader
from pruning_utils import *
from prune_model import *

import logging
import time
import copy

def load_pretrained_model(args, logger):
    """
    Load the pruned model from the path
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model 
    logger.info("=> evaluating pretrained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=(not args.random))

    # modify the model for cifar10
    if args.arch.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.arch.startswith('vgg'):
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    print('dataparallel mode')
    model = torch.nn.DataParallel(model).cuda()


    # Get the cifar10 dataset 
    train_loader, val_loader = get_cifar_dataloader(args)
    
    # The pretrained model's path
    pretrained_model_path = os.path.join('pretrained_models', 'cifar10-pretrain_model.pth.tar')
    # Load the pretrained model
    if os.path.isfile(pretrained_model_path):
        logger.info("=> loading pretrained model '{}'".format(pretrained_model_path))
    else:
        logger.info("=> no pretrained model found at '{}'".format(pretrained_model_path))
        return

    model.load_state_dict(torch.load(pretrained_model_path))

    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()

    # evaluate the model sparsity
    sparsity = check_sparsity(model)
    logger.info("=> model sparsity: {}".format(sparsity))

    # evaluate the model accuracy
    accuracy = validate(val_loader, model, criterion, args)
    logger.info("=> model accuracy: {}".format(accuracy))
    return model

def load_pruned_model(args, logger, prun_iter=0):
    """
    Load the pruned model from the path
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model 
    logger.info("=> evaluating pruned model '{}', at prune iteration '{}'".format(args.arch, prun_iter))
    model = models.__dict__[args.arch](pretrained=(not args.random))

    # modify the model for cifar10
    if args.arch.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.arch.startswith('vgg'):
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    print('dataparallel mode')
    model = torch.nn.DataParallel(model).cuda()

    # Get the cifar10 dataset 
    train_loader, val_loader = get_cifar_dataloader(args)

    pruned_model_name = str(prun_iter)+'model_best.pth.tar'

    # The pruned model's path
    pruned_model_path = os.path.join(args.save_dir, pruned_model_name)
    # Load the pretrained model
    if os.path.isfile(pruned_model_path):
        logger.info("=> loading pretrained model '{}'".format(pruned_model_path))
    else:
        logger.info("=> no pretrained model found at '{}'".format(pruned_model_path))
        return

    checkpoint = torch.load(pruned_model_path)
    args.start_epoch = checkpoint['epoch']
    args.start_state = checkpoint['state']
    if_pruned = checkpoint['if_pruned']

    if if_pruned:
        prune_model_custom(model.module, checkpoint['mask'], False)

    model.module.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()

    # evaluate the model sparsity
    sparsity = check_sparsity(model)
    logger.info("=> model sparsity: {}".format(sparsity))

    # evaluate the model accuracy
    accuracy = validate(val_loader, model, criterion, args)
    logger.info("=> pruned model accuracy: {}".format(accuracy))
    return model


if __name__ == '__main__':
    args = get_args()
    
    # Set up logging and logging directory and logging file using the current time
    logging.basicConfig(level=logging.INFO, format='%(message)s', 
                        handlers=[logging.FileHandler(os.path.join('logs/', time.strftime("%Y%m%d-%H%M%S") + '.log')), 
                        logging.StreamHandler()])
    
    logger = logging.getLogger(__name__)
    logger.info("Test the load model function")
    logger.info(args)

    # load_pretrained_model(args, logger)
    load_pruned_model(args, logger, prun_iter=10)

    
    

