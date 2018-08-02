import argparse
import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from modules.prune import prune_vanilla_elementwise
from modules.utils import get_sparsity, AverageMeter, Logger

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Pruning Sensitivity Scan')
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
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model: '
                         'pytorch: use pytorch official | '
                         'path to self-trained moel')
parser.add_argument('--pretrained-parallel', dest='pretrained_parallel',
                    action='store_true',
                    help='self-trained model starts with torch.nn.DataParallel')

parser.add_argument('--relatively-prune', dest='relatively', action='store_true',
                    help='relatively prune')


def main():
    global scan_log, rule_log
    args = parser.parse_args()

    dir_name = args.arch + '_' + datetime.datetime.now().strftime('%m%d_%H%M')
    log_dir = os.path.join('logs', os.path.join('scan', dir_name))
    os.makedirs(log_dir)
    scan_log = Logger(os.path.join(log_dir, 'scan.log'))
    rule_log = Logger(os.path.join(log_dir, 'recommend.rule'))

    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.pretrained == 'pytorch':
        print("=> using pre-trained model from pytorch model zoo")
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

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch.startswith('inception'):
        input_size = 299
    else:
        input_size = 224

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # test baseline top1 accuracy
    top1, _ = validate(val_loader=val_loader, model=model, sparsity=0)
    # sensitivity scan
    sensitivity_scan(model=model, val_loader=val_loader, relatively=args.relatively, top1=top1)

    scan_log.close()
    rule_log.close()


def validate(val_loader, model, sparsity):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

        print(' * Sparsity: {spars:.2f} | Prec@1: {top1.avg:.3f} | Prec@5: {top5.avg:.3f}'
              .format(spars=sparsity, top1=top1, top5=top5))

    return top1.avg, top5.avg


def sensitivity_scan(model, val_loader, top1, relatively=False):
    c1 = 100 - (100-top1) * 1.01
    c5 = 100 - (100-top1) * 1.05
    for i, (param_name, param) in enumerate(model.named_parameters()):
        print("{:3d} -> {param_name:^30} -> {param_shape}"
              .format(i, param_name=param_name, param_shape=param.size()))
        scan_log.write(content="@Param: {param_name:^30}".format(param_name=param_name), wrap=True)
        if param.dim() > 1:
            p1s, p5s = 1.0, 1.0
            scan_log.write(content="------ scanning param ------", wrap=True, verbose=True)
            param_clone = param.data.clone()
            origin_sparsity = get_sparsity(param=param_clone)
            for sparsity in np.arange(start=0.1, stop=1.0, step=0.1):
                if relatively:
                    sparsity *= origin_sparsity
                prune_vanilla_elementwise(sparsity=sparsity, param=param.data)
                top1, top5 = validate(val_loader=val_loader, model=model, sparsity=sparsity)
                scan_log.write(content="{spars:.3f}\t{top1:.3f}\t{top5:.5f}"
                                  .format(spars=sparsity, top1=top1, top5=top5),
                                  wrap=True)
                param.data.copy_(param_clone)
                if top1 > c5:
                    p5s = min(p5s, sparsity)
                    if top1 > c1:
                        p1s = min(p1s, sparsity)
                        break
            scan_log.flush()
            rule_log.write("{param_name} {stage_0:.5f},{stage_1:.5f}"
                           .format(param_name=param_name, stage_0=p1s, stage_1=p5s),
                           wrap=True, verbose=True, flush=True)
        else:
            scan_log.write("------ skipping param ------", wrap=True, verbose=True, flush=True)


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
