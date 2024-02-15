import torchvision.models as models
import argparse

def get_args():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    ############################# settings for pruning ###################################
    parser.add_argument('--prune-method', choices=['admm-ntl', 'admm-lda', 'original'], 
                        default='admm-lda', 
                        help='pruning method for model non-transferability')
    
    parser.add_argument('--image-size', default=32, type=int, help='image size')

    ############################# required settings ################################
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg11',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('--source', choices=['mnist', 'svhn', 'usps', 'mnistm', 'syn', 'cifar10', 'cifar100', 'stl'], 
                        default='mnist', 
                        help='source dataset')
    parser.add_argument('--target', choices=['mnist', 'svhn', 'usps', 'mnistm', 'syn', 'cifar10', 'cifar100', 'stl'],
                        default='usps', 
                        help='target dataset')
    parser.add_argument('--finetune-ratio', default=0.1, type=float, help='finetune ratio')
    parser.add_argument('--rho', default=1e-2, type=float, help='rho of admm')
    parser.add_argument('--alpha', default=1e-3, type=float, help='alpha of admm')
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


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)