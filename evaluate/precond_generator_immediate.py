"""
    This script runs whole benchmark set only for the purpose of securing against numerical bugs by generating preconditions
    It then records the detailed status and running time statistics
"""
import os
import time
import pickle
import argparse

import json
import numpy as np
import torch

from interp.interp_module import load_onnx_from_file
from interp.interp_utils import PossibleNumericalError
from interp.specified_vars import nodiff_vars

from trigger.inference.precondition_gen import PrecondGenModule, precondition_gen

# should be grist and/or debar
run_benchmarks = ['grist']

whitelist = []
blacklist = []

# impose the precondition on the input of immediately vulnerable operator
outtxtpath = 'empirical_study/precond/immediate'

if __name__ == '__main__':

    global_tot_bugs = 0
    global_tot_success = 0

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
            barefilename = file.rsplit('.', maxsplit=1)[0]
            if len(whitelist) > 0 and barefilename not in whitelist: continue
            if barefilename in blacklist: continue

            model = load_onnx_from_file(os.path.join(nowdir, file),
                                        customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
            bare_name = os.path.join(nowdir, file).split('/')[-1].split('.')[0]
            errors = model.analyze(model.gen_abstraction_heuristics(bare_name.split('.')[0]), {'average_pool_mode': 'coarse', 'diff_order': 0})

            fixes = []

            for err_node_name, err_details in errors.items():
                excep_objlist, root_cause, catastro = err_details
                print('err node name:', err_node_name)
                new_l, new_u = -np.inf, +np.inf
                div_fix_needed = False
                for excep_obj in excep_objlist:
                    if excep_obj.optype == 'Exp':
                        new_u = np.minimum(new_u, PossibleNumericalError.OVERFLOW_D)
                    elif excep_obj.optype == 'Log':
                        new_l = np.maximum(new_l, PossibleNumericalError.UNDERFLOW_LIMIT)
                    elif excep_obj.optype == 'Div':
                        div_fix_needed = True
                    elif excep_obj.optype == 'LogSoftMax':
                        new_l = np.maximum(new_l, 1e-17)
                        new_u = np.minimum(new_u, 1e+18)
                    elif excep_obj.optype == 'Sqrt':
                        new_l = np.maximum(new_l, PossibleNumericalError.UNDERFLOW_LIMIT)
                    else:
                        raise Exception('Unsupported fix for ' + excep_obj.optype + ' node yet.')

                    print(excep_obj.optype)
                    print(excep_obj.cur_range)
                if div_fix_needed:
                    fixcond = f'{err_node_name}.input <= {-PossibleNumericalError.UNDERFLOW_LIMIT} or {err_node_name}.input >= {PossibleNumericalError.UNDERFLOW_LIMIT}'
                elif new_l > -np.inf and new_u == np.inf:
                    fixcond = f'{err_node_name}.input >= {new_l}'
                elif new_u < np.inf and new_l == -np.inf:
                    fixcond = f'{err_node_name}.input <= {new_u}'
                else:
                    fixcond = f'{new_l} <= {err_node_name}.input <= {new_u}'
                now_fix_item = [err_node_name, excep_obj.optype, fixcond]
                fixes.append(now_fix_item)

            if not os.path.exists(outtxtpath):
                os.makedirs(outtxtpath)
            with open(os.path.join(outtxtpath, f'{bare_name}_ranum.txt'), 'w') as f:
                for fix in fixes:
                    f.write(f'Impose fix on node {fix[0]} with optype {fix[1]}:\n' + fix[2] + '\n\n')


    # print(f'{global_tot_bugs} bugs, {global_tot_success} out of them succeed to secure')
    # print('success cases:', success_cases)
    # print('failed cases:', failed_cases)
    # print('Unsupported Ops:', global_unsupported_ops)
    # print('Tot Cases:', len(average_shrink))
    # print('Mean Shrinkage:', np.mean(average_shrink))