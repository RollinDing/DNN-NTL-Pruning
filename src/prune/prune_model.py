import argparse
import os
import random
import shutil
import time
import warnings
import copy 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset
import numpy as np

from pruning_utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
############################# required settings ################################
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--decreasing_lr', default='10,20', help='decreasing strategy')
parser.add_argument('--save_dir', default=None, type=str)
parser.add_argument('--percent', default=0.2, type=float, help='pruning rate for each iteration')
parser.add_argument('--states', default=19, type=int, help='number of iterative pruning states')
parser.add_argument('--start_state', default=0, type=int, help='number of iterative pruning states')
parser.add_argument('--random', action="store_true", help="using random-init model")

############################# other settings ################################
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--pretrained', default=True, type=bool, help='pretrained model path')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')



best_acc1 = 0
best_epoch = 0

def main():
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                    'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1, best_epoch
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> using model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=(not args.random))

    # modify the model for cifar10
    if args.arch.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.arch.startswith('vgg'):
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    if_pruned = False

    # init pretrianed weight 
    ticket_init_weight = copy.deepcopy(model.state_dict())

    print('dataparallel mode')
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    # Data loading code for cifar10 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(32),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(32),
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
    subset_size = 45000 # for example, 5000 samples

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

    pretrained_model_path = os.path.join('pretrained_models', 'cifar10-pretrain_model.pth.tar')
    if not os.path.exists(pretrained_model_path):
        pretrain(train_loader, val_loader, model, criterion, optimizer, args)
    else:
        # Load the checkpoint
        model.load_state_dict(torch.load(pretrained_model_path))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            args.start_state = checkpoint['state']
            best_acc1 = checkpoint['best_acc1']
            if_pruned = checkpoint['if_pruned']
            ticket_init_weight = checkpoint['init_weight']

            if if_pruned:
                prune_model_custom(model.module, checkpoint['mask'], False)

            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}', pretrain the model and save it".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for prun_iter in range(args.start_state, args.states):
        check_sparsity(model.module, False)
        for epoch in range(args.start_epoch, args.epochs):
            if prun_iter == 0:

                acc1 = validate(val_loader, model, criterion, args)
                best_acc1 = acc1
                best_epoch = 0
                break 

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                best_epoch = epoch+1

            if if_pruned:
                mask_dict = extract_mask(model.state_dict())
            else:
                mask_dict = None

            save_checkpoint({  
                'epoch': epoch + 1,
                'state': prun_iter,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'mask': mask_dict,
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'if_pruned': if_pruned,
                'init_weight':ticket_init_weight
            }, is_best, checkpoint=args.save_dir, best_name=str(prun_iter)+'model_best.pth.tar')

        check_sparsity(model.module, False)
        print('best TA = ', best_acc1, 'best epoch = ', best_epoch)

        # start pruning 
        print('start pruning model')
        pruning_model(model.module, args.percent, False)
        if_pruned = True

        current_mask = extract_mask(model.state_dict())
        remove_prune(model.module, False)

        #weight rewind 
        model.module.load_state_dict(ticket_init_weight)
        best_acc1 = 0 
        best_epoch = 0

        prune_model_custom(model.module, current_mask, False)
        validate(val_loader, model, criterion, args)
        args.start_epoch=0
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

def pretrain(train_loader, val_loader, model, criterion, optimizer, args):
    print('pretrain the model')
    best_acc1 = 0
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    for epoch in range(30):
        print('Current Learning Rate is ', optimizer.state_dict()['param_groups'][0]['lr'])
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            # save checkpoint
            torch.save(model.state_dict(), os.path.join('pretrained_models', 'cifar10-pretrain_model.pth.tar'))
    
        scheduler.step()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', best_name='model_best.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best_name))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            # use .reshape
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()