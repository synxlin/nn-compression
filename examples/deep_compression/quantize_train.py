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

from modules.quantize import Quantizer
from modules.utils import AverageMeter, Logger

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
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR',
                    help='initial learning rate (default: 0.001 |'
                         ' for inception recommend 0.0256)')
parser.add_argument('--lr-decay', default=0.1, type=float, metavar='LD',
                    help='every N1,N2,... epochs learning rate decays by LD '
                         '(default:0.1 | for inception recommend 0.16)')
parser.add_argument('--lr-decay-step', default=5, metavar='N', type=int,
                    help='every N epochs learning rate decays (default: 5)')
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
parser.add_argument('--pretrained', dest='pretrained', default="",
                    help='use pre-trained model (after nn.DataParallel)')
parser.add_argument('--quantize-rule', default="", help='path to quantization rule file')
parser.add_argument('-z', '--not-fix-zeros', dest='not_fix_zeros', default=False,
                    action="store_true", help='not fix zeros in quantization')
parser.add_argument('-c', '--update-centers-only', dest='update_centers', default=False,
                    action="store_true", help='update centers of codebook per iteration')
parser.add_argument('-l','--update-centers-and-labels', dest='update_labels', default=False,
                    action="store_true", help='update centers of codebook and labels per iteration')
parser.add_argument('-r','--re-quantize', dest='re_quantize', default=False,
                    action="store_true", help='re-quantize (re-kmeans) per iteration')
parser.add_argument('--nGPU', type=int, default=4, help='gpus number to use')
parser.add_argument('--visdom', dest='visdom', default=False, action='store_true',
                    help='open visualization')

best_prec1 = 0


def main():
    global args, best_prec1, train_log, test_log, quantizer
    args = parser.parse_args()
    if not args.update_centers and not args.update_labels and not args.re_quantize:
        vars(args)['update_centers'] = True
    dir_name = args.arch + '_' + datetime.datetime.now().strftime('%m%d_%H%M')
    log_dir = os.path.join('logs', os.path.join('quantize', dir_name))
    checkpoint_dir = os.path.join('checkpoints', os.path.join('quantize', dir_name))
    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)
    train_log = Logger(os.path.join(log_dir, 'train.log'))
    test_log = Logger(os.path.join(log_dir, 'test.log'))
    config_log = Logger(os.path.join(log_dir, 'config.log'))
    vars(args)['experiment_id'] = dir_name
    for key, val in vars(args).items():
        config_log.write('{}: {}\n'.format(key, val))

    loss_results, top1_results, top5_results = torch.FloatTensor(args.epochs), torch.FloatTensor(args.epochs), \
                                               torch.FloatTensor(args.epochs)
    if args.visdom:
        from visdom import Visdom
        viz = Visdom()
        opts = [dict(title=args.experiment_id + ' Loss', ylabel='Loss', xlabel='Epoch'),
                dict(title=args.experiment_id + ' Top-1', ylabel='Top-1', xlabel='Epoch'),
                dict(title=args.experiment_id + ' Top-5', ylabel='Top-5', xlabel='Epoch')]
        viz_windows = [None, None, None]
        epochs = torch.arange(0, args.epochs)

    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.pretrained == 'True':
            print("=> using pre-trained model from model zoo")
            model = models.__dict__[args.arch](pretrained=True)
    else:
        if args.arch.startswith('inception'):
            model = models.__dict__[args.arch](transform_input=True)
        else:
            model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features, list(range(args.nGPU)))
        model.cuda()
    else:
        model = torch.nn.DataParallel(model, list(range(args.nGPU))).cuda()

    if args.pretrained and args.pretrained != 'True':
        print("=> using pre-trained model '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])

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

    # quantize
    quantizer = Quantizer(args.quantize_rule, fix_zeros=(not args.not_fix_zeros))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if 'best_prec1' in checkpoint:
                best_prec1 = checkpoint['best_prec1']
            is_load_optimizer = False
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                is_load_optimizer = True
                optimizer.zero_grad()
            is_load_codebooks = False
            if 'codebooks' in checkpoint:
                quantizer.load_codebooks(checkpoint['codebooks'])
                is_load_codebooks = True
            loss_results, top1_results, top5_results = checkpoint['loss_results'], checkpoint['top1_results'], \
                                                       checkpoint['top5_results']
            # Add previous scores to visdom graph
            if args.visdom and loss_results is not None:
                x_axis = epochs[0:args.start_epoch]
                y_axis = [loss_results[0:args.start_epoch], top1_results[0:args.start_epoch],
                          top5_results[0:args.start_epoch]]
                for x in range(len(viz_windows)):
                    viz_windows[x] = viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        opts=opts[x],
                    )
            print("=> loaded checkpoint (epoch {}, best_prec1 {}, is_load_optimizer {}, is_load_codebooks {})"
                  .format(args.start_epoch, best_prec1, is_load_optimizer, is_load_codebooks))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

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
            transforms.Scale(int(input_size/0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    quantizer.quantize_model(model=model)
    best_prec1, _ = validate(val_loader, model, criterion, -1)
    save_checkpoint({
        'epoch': 0,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'codebooks': quantizer.get_codebooks(),
    }, False, filename='quantize.pth.tar', checkpoint_dir=checkpoint_dir)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch, val_loader)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, epoch)

        loss_results[epoch] = loss
        top1_results[epoch] = prec1
        top5_results[epoch] = prec5

        if args.visdom:
            x_axis = epochs[0:epoch + 1]
            y_axis = [loss_results[0:epoch + 1], top1_results[0:epoch + 1],
                      top5_results[0:epoch + 1]]
            for x in range(len(viz_windows)):
                if viz_windows[x] is None:
                    viz_windows[x] = viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        opts=opts[x],
                    )
                else:
                    viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        win=viz_windows[x],
                        update='replace',
                    )

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'codebooks': quantizer.get_codebooks(),
            'loss_results': loss_results,
            'top1_results': top1_results,
            'top5_results': top5_results,
        }, is_best, checkpoint_dir=checkpoint_dir)


def train(train_loader, model, criterion, optimizer, epoch, val_loader):
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

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        if args.arch.startswith('inception'):
            output, aux_output = model(input_var)
            loss = criterion(output, target_var) + criterion(aux_output, target_var)

        else:
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # quantize
        quantizer.quantize_model(model, update_centers=args.update_centers,
                                 update_labels=args.update_labels, re_quantize=args.re_quantize)

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
        if i == len(train_loader) // 2:
            validate(val_loader, model, criterion, epoch)
            model.train()
    print('* Train epoch # %d    top1:  %.3f  top5:  %.3f' % (epoch, top1.avg, top5.avg))
    train_log.write('%d\t%.4e\t%.4e\t%.4e\t\n' % (epoch, top1.avg, top5.avg, losses.avg))
    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
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

    print('* Test epoch # %d    top1: %.3f    top5: %.3f' % (epoch, top1.avg, top5.avg))
    test_log.write('%d\t%.4e\t%.4e\n' % (epoch, top1.avg, top5.avg))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay_step epochs"""
    decay = epoch // args.lr_decay_step
    lr = args.lr * (args.lr_decay ** decay)
    print('Epoch: %d  change learning rate to %f = origin time 0.1 ** %d' % (epoch,
                                                                             lr, decay))
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