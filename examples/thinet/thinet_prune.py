import argparse
import datetime
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from modules.utils import AverageMeter, Logger
from .vgg_pruner import VGGPruner

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("vgg")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ThiNet Pruning')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: vgg16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay-step', default=4, type=int, metavar='N',
                    help='every N epochs lr decays by 0.1 (default:4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pruning-rule', default='',
                    help='path to quantization rule file')
parser.add_argument('--method', default='greedy', type=str, metavar='METHOD',
                    help='channel selection method in ThiNet Pruning:' +
                         ' | '.join(['greedy', 'random']) +
                         ' (default: greedy)')
parser.add_argument('--rb', '--reconstruction-batch-size', default=128,
                    type=int, metavar='N', dest='rcn_batch_size',
                    help='mini-batch size for ThiNet Pruning '
                         'Reconstruction (default: 128)')
parser.add_argument('--rcn-gpu', dest='rcn_gpu', action='store_true',
                    help='use gpu to perform ThiNet Weight Reconstruction')

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
    prune_log = Logger(os.path.join(log_dir, 'prune.log'))
    config_log = Logger(os.path.join(log_dir, 'config.log'))

    for k, v in vars(args).items():
        config_log.write(content="{k} : {v}".format(k=k, v=v), wrap=True, flush=True)
    config_log.close()

    # create model
    print("=" * 89)
    print("=> creating model '{}'".format(args.arch))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model = checkpoint['model']
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    elif args.pretrained:
        print("=> using pre-trained model from model zoo")
        model = models.__dict__[args.arch](pretrained=True)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = models.__dict__[args.arch]()
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not args.resume:
        rcn_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.rcn_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        prune(train_loader=train_loader, val_loader=val_loader, rcn_loader=rcn_loader,
              model=model, criterion=criterion)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(lr_decay_step=args.lr_decay_step,
                             optimizer=optimizer, epoch=epoch)

        # train for one epoch
        train(train_loader=train_loader, model=model, criterion=criterion,
              optimizer=optimizer, epoch=epoch, log=True)

        # evaluate on validation set
        prec1 = validate(val_loader=val_loader, model=model,
                         criterion=criterion, epoch=epoch, log=True)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'model': model,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best=is_best, checkpoint_dir=checkpoint_dir)


def train(train_loader, model, criterion, optimizer, epoch, log=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    print("=" * 89)
    print(' * Train Epoch: {epoch:3d} | Prec@1: {top1.avg:.3f} | Prec@5: {top5.avg:.3f}'
          .format(epoch=epoch, top1=top1, top5=top5))
    print("=" * 89)
    if log:
        train_log.write(content="{epoch}\t"
                                "{top1.avg:.4e}\t"
                                "{top5.avg:.4e}\t"
                                "{loss.avg:.4e}"
                        .format(epoch=epoch, top1=top1, top5=top5, loss=losses), wrap=True, flush=True)


def validate(val_loader, model, criterion, epoch, log=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

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
        if log:
            test_log.write(content="{epoch}\t"
                                   "{top1.avg:.4e}\t"
                                   "{top5.avg:.4e}\t"
                           .format(epoch=epoch, top1=top1, top5=top5), wrap=True, flush=True)

    return top1.avg


def prune(train_loader, val_loader, rcn_loader, model, criterion):
    print("=" * 89)
    origin_prec1 = validate(val_loader=val_loader, model=model, criterion=criterion, epoch=0)

    input_iter = iter(rcn_loader)

    print("=" * 89)
    print("start ThiNet Pruning")
    pruner = VGGPruner(rule=args.pruning_rule)
    prune_inputs = pruner.get_prune_inputs(model=model)
    for (module_name, module, next_module,
         fn_input_feature, fn_next_input_feature) in prune_inputs:
        input, _ = input_iter.__next__()
        pruner.prune_module(module_name=module_name, module=module,
                            next_module=next_module, fn_input_feature=fn_input_feature,
                            fn_next_input_feature=fn_next_input_feature,
                            input=input, method=args.method, cpu=(not args.rcn_gpu),
                            verbose=True)
        prec1 = validate(val_loader=val_loader, model=model, criterion=criterion, epoch=0)
        if prec1 > origin_prec1:
            continue
        print("=" * 89)
        print("Fine-tuning")
        print("=" * 89)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        train(train_loader=train_loader, model=model,
              criterion=criterion, optimizer=optimizer, epoch=0)
        adjust_learning_rate(lr_decay_step=1, optimizer=optimizer, epoch=1)
        train(train_loader=train_loader, model=model,
              criterion=criterion, optimizer=optimizer, epoch=1)
        del optimizer
        validate(val_loader=val_loader, model=model,
                 criterion=criterion, epoch=0)
    print("=" * 89)
    print("stop ThiNet Pruning")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename, pickle_protocol=4)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def adjust_learning_rate(lr_decay_step, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay_step epochs"""
    lr = args.lr * (0.1 ** (epoch // lr_decay_step))
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
