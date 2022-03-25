"""
    This script nicely print the preconditions
"""

import os
import json
import numpy as np
import torch

from interp.interp_module import load_onnx_from_file
from evaluate.seeds import seeds



max_iter = 1000
center_lr = 0.1
scale_lr = 0.1
min_step = 0.01

precond_code = f'iter_{max_iter}_lr_{center_lr}_{scale_lr}_minstep_{min_step}'


if __name__ == '__main__':

    ordering = os.listdir('model_zoo/grist_protobufs_onnx')
    ordering = [x[:-5] for x in ordering if x.endswith('.onnx')]
    ordering = sorted(ordering, key=lambda x: int(''.join([y for y in x if '0' <= y <= '9'])) +
                                              sum([0.01 * ord(y) - 0.01 * ord('a') for y in x if 'a' <= y <= 'z']))

    ans = [[x] + ['' for _ in range(13)] for x in ordering]

    for mode in ['all', 'weight', 'input']:
        for i, mname in enumerate(ordering):


            modelpath = f'model_zoo/grist_protobufs_onnx/{mname}.onnx'
            model = load_onnx_from_file(modelpath,
                                        customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})

            bare_name = modelpath.split('/')[-1].split('.')[0]
            model.analyze(model.gen_abstraction_heuristics(bare_name.split('.')[0]), {'average_pool_mode': 'coarse', 'diff_order': 0})

            with open(f'results/precond_gen/grist/all/{mode}/{precond_code}/{mname}.json', 'r') as f:
                data = json.load(f)
                if data['success_cnt'] > 0:
                    results = torch.load(f'results/precond_gen/grist/all/{mode}/{precond_code}/{mname}_data.pt')
                    out_path = f'empirical_study/precond/{mode}'
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    with open(os.path.join(out_path, f'{mname}_debarus.txt'), 'w') as f:
                        f.write(json.dumps(data['precond_stat'], indent=2) + '\n\n')
                        for node, bounds in results.items():
                            if node not in model.initial_abstracts: continue
                            if torch.sum(bounds.ub - bounds.lb) < 1e-5:
                                if bounds.lb.numel() > 1:
                                    f.write(f'{node} = {bounds.lb.data.numpy()}\n\n')
                                elif bounds.lb.numel() == 1:
                                    f.write(f'{node} = {bounds.lb.item()}\n\n')
                            else:
                                if bounds.lb.numel() > 1:
                                    f.write(f'{bounds.lb.data.numpy()} <= {node} <= {bounds.ub.data.numpy()}\n\n')
                                else:
                                    f.write(f'{bounds.lb.item()} <= {node} <= {bounds.ub.item()}\n\n')

                        f.write('=' * 10 + 'initial bounds' + '=' * 10 + '\n\n')
                        for node in results:
                            if node not in model.initial_abstracts: continue
                            bounds = model.initial_abstracts[node]
                            if torch.sum(bounds.ub - bounds.lb) < 1e-5:
                                if bounds.lb.numel() > 1:
                                    f.write(f'{node} = {bounds.lb.data.numpy()}\n\n')
                                elif bounds.lb.numel() == 1:
                                    f.write(f'{node} = {bounds.lb.item()}\n\n')
                            else:
                                if bounds.lb.numel() > 1:
                                    f.write(f'{bounds.lb.data.numpy()} <= {node} <= {bounds.ub.data.numpy()}\n\n')
                                else:
                                    f.write(f'{bounds.lb.item()} <= {node} <= {bounds.ub.item()}\n\n')

