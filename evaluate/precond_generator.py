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

from trigger.inference.precondition_gen import PrecondGenModule, precondition_gen

MAX_ITER = 100

# should be grist and/or debar
run_benchmarks = ['grist']


# whitelist = ['17']
whitelist = []
blacklist = []
# blacklist = ['17']

average_shrink = list()

goal = 'all'
# variables = 'all'
# variables = 'input'
variables = 'weight'

max_iter = 1000
# center_lr = 0.1
# scale_lr = 0.1
# min_step = 0.1
center_lr = 0.01
scale_lr = 0.01
min_step = 0.01

if __name__ == '__main__':
    global_unsupported_ops = dict()

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

            tot_success, tot_bugs, result_details, running_times, running_iters = precondition_gen(os.path.join(nowdir, file), goal, variables, debug=False,
                                                                                                   max_iter=max_iter, center_lr=center_lr, scale_lr=scale_lr, min_step=min_step)

            pkl_package = {'time_stat': running_times, 'iter_stat': running_iters, 'numerical_bugs': tot_bugs, 'success_cnt': tot_success,
                           'precond_stat': result_details}
            print(pkl_package)

            pkl_path = f'results/precond_gen/{bench_type}/{goal}/{variables}/iter_{max_iter}_lr_{center_lr}_{scale_lr}_minstep_{min_step}/{file[:-5]}.pkl'
            if not os.path.exists(os.path.dirname(pkl_path)):
                os.makedirs(os.path.dirname(pkl_path))
            with open(pkl_path, 'wb') as f:
                pickle.dump(pkl_package, f)
            print(f'saved to {pkl_path}')

            global_tot_bugs += tot_bugs
            global_tot_success += tot_success

            for det in result_details.values():
                average_shrink.append(det['average_shrinkage'])
                print('now avg shrinkage:', det['average_shrinkage'])

    print(f'{global_tot_bugs} bugs, {global_tot_success} out of them succeed to secure')
    print('Unsupported Ops:', global_unsupported_ops)
    print('Tot Cases:', len(average_shrink))
    print('Mean Shrinkage:', np.mean(average_shrink))