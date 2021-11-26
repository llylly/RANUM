"""
    This script runs whole benchmark set only for the purpose of securing against numerical bugs by generating preconditions
    It then records the detailed status and running time statistics
"""
import os
import time
import pickle

import numpy as np
import torch

from interp.interp_module import load_onnx_from_file
from interp.interp_utils import AbstractionInitConfig
from interp.specified_vars import nodiff_vars

from trigger.inference.precondition_gen import PrecondGenModule

MAX_ITER = 100

# should be grist and/or debar
run_benchmarks = ['grist']

average_shrink = list()

if __name__ == '__main__':
    global_unsupported_ops = dict()
    for bench_type in run_benchmarks:
        if bench_type == 'grist':
            print('on GRIST bench')
            nowdir = 'model_zoo/grist_protobufs_onnx'
        elif bench_type == 'debar':
            print('on DEBAR bench')
            nowdir = 'model_zoo/tf_protobufs_onnx'
        files = sorted([x for x in os.listdir(nowdir) if x.endswith('.onnx')])
        nowlen = len(files)
        for id, file in enumerate(files):
            print(f'[{id+1} / {nowlen}] {file}')
            stime = time.time()

            model = load_onnx_from_file(os.path.join(nowdir, file),
                                        customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
            loadtime = time.time()

            res = model.analyze(model.gen_abstraction_heuristics(file), {'average_pool_mode': 'coarse'})
            analyzetime = time.time()

            runningtime_stat = {'load': loadtime - stime, 'analyze': analyzetime - loadtime}
            bugs = res
            unsupported_ops = list(model.unimplemented_types)

            if len(unsupported_ops) > 0:
                global_unsupported_ops[f'{bench_type}/{file}'] = unsupported_ops
                print(f'* Unsupported ops ({len(unsupported_ops)}): {unsupported_ops}')


            if len(bugs) > 0:
                print(f'* Found {len(bugs)} numerical bugs')
                # securing all error points
                err_nodes, err_exceps = list(), list()
                for k, v in bugs.items():
                    error_entities, root, catastro = v
                    if not catastro:
                        for error_entity in error_entities:
                            err_nodes.append(error_entity.var_name)
                            err_exceps.append(error_entity)

                success = False

                precond_module = PrecondGenModule(model, nodiff_vars)
                # I only need the zero_grad method from an optimizer, therefore any optimizer works
                optimizer = torch.optim.Adam(precond_module.parameters(), lr=0.1)

                # for kk, vv in precond_module.abstracts.items():
                #     print(kk, 'lb     :', vv.lb)
                #     print(kk, 'ub     :', vv.ub)

                for iter in range(MAX_ITER):
                    # print('----------------')
                    optimizer.zero_grad()
                    loss, errors = precond_module.forward(err_nodes, err_exceps)
                    # for kk, vv in precond_module.abstracts.items():
                    #     try:
                    #         vv.lb.retain_grad()
                    #         vv.ub.retain_grad()
                    #     except:
                    #         print(kk, 'cannot retain grad')

                    print('iter =', iter, 'loss =', loss, '# errors =', len(errors))

                    loss.backward()

                    # for kk, vv in precond_module.abstracts.items():
                    #     print(kk, 'lb grad:', vv.lb.grad)
                    #     print(kk, 'ub grad:', vv.ub.grad)
                    #     print(kk, 'lb     :', vv.lb)
                    #     print(kk, 'ub     :', vv.ub)

                    precond_module.grad_step()
                    if len(errors) == 0:
                        success = True
                        print('securing condition found!')
                        break

                # for kk, vv in precond_module.abstracts.items():
                #     print(kk, 'lb     :', vv.lb)
                #     print(kk, 'ub     :', vv.ub)
                #     print(kk, 'lb grad:', vv.lb.grad)
                #     print(kk, 'ub grad:', vv.ub.grad)

                print('--------------')
                runningtime_stat['precond'] = time.time() - stime
                print(f'* Time: load - {runningtime_stat["load"]:.3f} s | analyze - {runningtime_stat["analyze"]:.3f} s')
                if success:
                    print('Success!')
                    precond_stat = precond_module.precondition_study()
                    print('Shrinkage', precond_stat['average_shrinkage'])
                    average_shrink.append(precond_stat['average_shrinkage'])
                else:
                    print('!!! Not success')
                    precond_stat = None
                    raise Exception('failed here :(')
                print('--------------')


                pkl_package = {'time_stat': runningtime_stat, 'numerical_bugs': bugs, 'unspported_ops': unsupported_ops,
                               'success': success, 'precond_stat': precond_stat}
                pkl_path = f'results/precond_gen/{bench_type}/{file[:-5]}.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(pkl_package, f)
                print(f'saved to {pkl_path}')

    print('Unsupported Ops:', global_unsupported_ops)
    print('Tot Cases:', len(average_shrink))
    print('Mean Shrinkage:', np.mean(average_shrink))