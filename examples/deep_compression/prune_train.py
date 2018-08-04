import argparse
import datetime
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from slender.prune import VanillaPruner
from slender.utils import AverageMeter, Logger

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--nGPU', type=int, default=4,
                    help='the number of gpus for training')

parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR',
                    help='initial learning rate (default: 0.001 |'
                         ' for inception recommend 0.0256)')
parser.add_argument('--lr-decay-step', default=15, type=int, metavar='N',
                    help='every N epochs learning rate decays (default:15)')
parser.add_argument('--lr-decay', default=0.1, type=float, metavar='LD',
                    help='every lr-decay-step epochs learning rate decays '
                         'by LD (default:0.1 | for inception recommend 0.16)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='WD', help='weight decay for sgd (default: 1e-4)')
parser.add_argument('--alpha', default=0.9, type=float,
                    metavar='ALPHA', help='alpha for RMSprop (default: 0.9)')
parser.add_argument('--eps', '--epsilon', default=1.0, type=float,
                    metavar='EPS', help='epsilon for RMSprop (default: 1.0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model: '
                         'pytorch: use pytorch official | '
                         'path to self-trained moel')
parser.add_argument('--pretrained-parallel', dest='pretrained_parallel',
                    action='store_true',
                    help='self-trained model starts with torch.nn.DataParallel')

parser.add_argument('--prune-rule', default='',
                    help='path to prune rule file')
parser.add_argument('--prune-stage', default=0, type=int, metavar='N',
                    help='pruning stage')


best_prec1 = 0


def main():
    global args, best_prec1, train_log, test_log
    args = parser.parse_args()

    dir_name = args.arch + '_' + datetime.datetime.now().strftime('%m%d_%H%M')
    log_dir = os.path.join('logs', os.path.join('prune', dir_name))
    checkpoint_dir = os.path.join('checkpoints', os.path.join('prune', dir_name))
    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)
    train_log = Logger(os.path.join(log_dir, 'train.log'))
    test_log = Logger(os.path.join(log_dir, 'test.log'))
    config_log = Logger(os.path.join(log_dir, 'config.log'))

    for k, v in vars(args).items():
        config_log.write(content="{k} : {v}".format(k=k, v=v), wrap=True, flush=True)
    config_log.close()

    # create model
    print("=" * 89)
    print("=> creating model '{}'".format(args.arch))

    if args.pretrained == 'pytorch':
        print("=> using pre-trained model from model zoo")
        model = models.__dict__[args.arch](pretrained=True)
        args.pretrained_parallel = False
    else:
        if args.arch.startswith('inception'):
            model = models.__dict__[args.arch](transform_input=True)
        else:
            model = models.__dict__[args.arch]()
        if args.pretrained and not args.pretrained_parallel:
            if os.path.isfile(args.pretrained):
                print("=> using pre-trained model '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features, device_ids=list(range(args.nGPU)))
        model.cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.nGPU))).cuda()

    if args.pretrained and args.pretrained_parallel:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.arch.startswith('inception'):
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                        alpha=args.alpha, eps=args.eps,
                                        momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    pruner = VanillaPruner(rule=args.prune_rule)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            best_prec1 = checkpoint['best_prec1']
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.zero_grad()
            pruner.load_state_dict(checkpoint['pruner'], replace_rule=False)
            print("=> loaded checkpoint (epoch {:3d}, best_prec1 {:.3f})"
                  .format(args.start_epoch, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("=" * 89)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch.startswith('inception'):
        input_size = 299
    else:
        input_size = 224

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        if epoch == 0:
            pruner.prune(model=model, stage=args.prune_stage, update_masks=True)
            best_prec1 = validate(val_loader, model, criterion, epoch)

        adjust_learning_rate(optimizer, epoch=epoch)

        # train for one epoch
        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer,
              pruner=pruner, epoch=epoch)

        # evaluate on validation set
        prec1 = validate(val_loader=val_loader, model=model, criterion=criterion, epoch=epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'pruner': pruner.state_dict(),
        }, is_best=is_best, checkpoint_dir=checkpoint_dir)
        if (epoch + 1) in args.prune_step:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'prec1': prec1,
                'pruner': pruner.state_dict(),
            }, is_best=False, filename='stage_{}.pth.tar'.format(stage_id),
                checkpoint_dir=checkpoint_dir)

    train_log.close()
    test_log.close()


def train(train_loader, model, criterion, optimizer, pruner, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    print("=" * 89)

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        if args.arch.startswith('inception'):
            output, aux_output = model(input)
            loss = criterion(output, target) + criterion(aux_output, target)

        else:
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pruning
        pruner.prune(model=model, stage=args.prune_stage, update_masks=False)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print("Epoch: [{0}][{1}/{2}]\t"
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                  "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                  "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    print("=" * 89)
    print(' * Train Epoch: {epoch:3d} | Prec@1: {top1.avg:.3f} | Prec@5: {top5.avg:.3f}'
          .format(epoch=epoch, top1=top1, top5=top5))
    print("=" * 89)
    train_log.write(content="{epoch}\t"
                            "{top1.avg:.4e}\t"
                            "{top5.avg:.4e}\t"
                            "{loss.avg:.4e}"
                    .format(epoch=epoch, top1=top1, top5=top5, loss=losses), wrap=True, flush=True)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    print("=" * 89)

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        print("=" * 89)
        print(' * Test Epoch: {epoch:3d} | Prec@1: {top1.avg:.3f} | Prec@5: {top5.avg:.3f}'
              .format(epoch=epoch, top1=top1, top5=top5))
        print("=" * 89)
        test_log.write(content="{epoch}\t"
                               "{top1.avg:.4e}\t"
                               "{top5.avg:.4e}\t"
                       .format(epoch=epoch, top1=top1, top5=top5), wrap=True, flush=True)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename, pickle_protocol=4)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch=0):
    """
    Sets the learning rate to the initial LR decayed by args.lr_decay every lr_decay_step epochs
    :param optimizer:
    :param epoch:
    :param stage:
    :return:
    """
    decay = epoch // args.lr_decay_step
    lr = args.lr * (args.lr_decay ** decay)
    print("Stage: {stage:2d}  Epoch: {epoch:3d} | "
          "learning rate = {lr:.6f} = origin x ({lr_decay:.2f} ** {decay:2d})"
          .format(stage=args.prune_stage, epoch=epoch, lr=lr, lr_decay=args.lr_decay, decay=decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
