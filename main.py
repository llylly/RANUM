import argparse

import graphparser.tf_adaptor as tf_adaptor
import graphparser.torch_adaptor as torch_adaptor

parser = argparse.ArgumentParser()
parser.add_argument('modelpath', type=str, help='model architecture file path')
parser.add_argument('--format', type=str, choices=['protobuf', 'torch'], default='protobuf', help='model architecture format')
parser.add_argument('--modelvar', type=str, help='for Torch model specify the variable name')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.format == 'protobuf':
        model = torch_adaptor.parseProtoBuf(args.modelpath)
    elif args.format == 'torch':
        if args.modelvar is not None:
            model = torch_adaptor.parseTorch(args.modelpath, args.modelvar)
        else:
            model = torch_adaptor.parseTorch(args.modelpath)

    print('model type is', type(model))