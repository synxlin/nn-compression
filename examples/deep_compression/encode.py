import argparse
import os
import datetime

import torch
import torchvision.models as models

from modules.coding import Codec
from modules.utils import Logger

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Encoding')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to self-trained moel')
parser.add_argument('--pretrained-parallel', dest='pretrained_parallel',
                    action='store_true',
                    help='self-trained model starts with torch.nn.DataParallel')
parser.add_argument('--coding-rule', default='',
                    help='path to coding rule file')


def main():
    args = parser.parse_args()

    dir_name = args.arch + '_' + datetime.datetime.now().strftime('%m%d_%H%M')
    log_dir = os.path.join('logs', os.path.join('coding', dir_name))
    checkpoint_dir = os.path.join('checkpoints', os.path.join('coding', dir_name))
    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)

    config_log = Logger(os.path.join(log_dir, 'config.log'))

    for k, v in vars(args).items():
        config_log.write(content="{k} : {v}".format(k=k, v=v), wrap=True, flush=True)
    config_log.close()

    print("=" * 89)
    print("=> creating model '{}'".format(args.arch))

    if args.arch.startswith('inception'):
        model = models.__dict__[args.arch](transform_input=True)
    else:
        model = models.__dict__[args.arch]()

    if args.pretrained:
        if args.pretrained_parallel:
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features).cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

            if os.path.isfile(args.pretrained):
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained)
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint")
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))

            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = model.features.module
            else:
                model = model.module

            model = model.cpu()
        else:
            if os.path.isfile(args.pretrained):
                print("=> using pre-trained model '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))

        codec = Codec(rule=args.coding_rule)

        encoded_model = codec.encode(model=model)

        torch.save({
            'state_dict': encoded_model.state_dict(),
        }, os.path.join(checkpoint_dir, 'encode.pth.tar'), pickle_protocol=4)

    else:
        print("=> no checkpoint")

    print("=" * 89)


if __name__ == '__main__':
    main()
