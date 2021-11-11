import argparse
import time
import os

import onnxruntime

from interp.interp_module import load_onnx_from_file
from interp.interp_utils import AbstractionInitConfig, PossibleNumericalError

stime = time.time()


def prompt(msg):
    print(f'[{time.time() - stime:.3f}s] ' + msg)


parser = argparse.ArgumentParser()
parser.add_argument('modelpath', type=str, help='model architecture file path')
parser.add_argument("--continue-prop", action='store_true', help='continue propagating after numerical error')

if __name__ == '__main__':
    args = parser.parse_args()

    model = load_onnx_from_file(args.modelpath,
                                customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572,
                                                 'unk__7156': 128,  # lfads.pbtxt.onnx
                                                 })
    prompt('model initialized')
    prompt(f'--continue-prop: {args.continue_prop}')
    PossibleNumericalError.continue_prop = args.continue_prop

    res = model.analyze(model.gen_abstraction_heuristics(os.path.split(args.modelpath)[-1].split('.')[0]),
                        {'average_pool_mode': 'coarse',
                         'onehot_mode': 'coarse'})
    prompt('analysis done')
    if len(res) == 0:
        print('No numerical bug')
    else:
        print(f'{len(res)} possible numerical bug(s)')
        for k, v in res.items():
            print(f'- On tensor {k} triggered by operator {v[1]}:')
            for item in v[0]:
                print(str(item))
