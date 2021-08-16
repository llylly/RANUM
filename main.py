import argparse

import graphparser.tf_adaptor as tf_adaptor
import graphparser.torch_adaptor as torch_adaptor

from interp.interp_module import load_onnx_from_file, analyze

parser = argparse.ArgumentParser()
parser.add_argument('modelpath', type=str, help='model architecture file path')

if __name__ == '__main__':
    args = parser.parse_args()


    model = load_onnx_from_file(args.modelpath)

    res = analyze(model)

    print(res)

