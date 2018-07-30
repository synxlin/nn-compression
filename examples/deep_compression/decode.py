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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained encoded model (before nn.DataParallel)')


def main():
    args = parser.parse_args()

    dir_name = args.arch + '_' + datetime.datetime.now().strftime('%m%d_%H%M')
    checkpoint_dir = os.path.join('checkpoints', os.path.join('coding', dir_name))
    os.makedirs(checkpoint_dir)

    print("=" * 89)
    print("=> creating model '{}'".format(args.arch))

    if args.arch.startswith('inception'):
        model = models.__dict__[args.arch](transform_input=True)
    else:
        model = models.__dict__[args.arch]()

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)

        model = Codec.decode(model=model, state_dict=checkpoint['state_dict'])

        torch.save({
            'state_dict': model.state_dict(),
        }, os.path.join(checkpoint_dir, 'decode.pth.tar'), pickle_protocol=4)

    else:
        print("=> no checkpoint")

    print("=" * 89)


if __name__ == '__main__':
    main()
