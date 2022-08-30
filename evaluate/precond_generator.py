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
variables = 'all'
# variables = 'input'
# variables = 'weight'

# max_iter = 100
# center_lr = 0.1
# scale_lr = 0.1
# min_step = 0.1

max_iter = 1000
center_lr = 0.1
scale_lr = 0.1
min_step = 0.01

approach = ''

success_cases = list()
failed_cases = list()

parser = argparse.ArgumentParser()
parser.add_argument('method', choices=['ranum', 'gd', 'ranumexpand'])
parser.add_argument('var', choices=['all', 'input', 'weight'], default='all')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.method == 'ranum':
        approach = ''
    elif args.method == 'gd':
        approach = 'gd'
    elif args.method == 'ranumexpand':
        approach = 'ranumexpand'
    variables = args.var

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

            pkl_path = f'results/precond_gen/{bench_type}/{goal}/{variables}/iter_{max_iter}_lr_{center_lr}_{scale_lr}_minstep_{min_step}{approach}/{file[:-5]}.pkl'
            json_path = f'results/precond_gen/{bench_type}/{goal}/{variables}/iter_{max_iter}_lr_{center_lr}_{scale_lr}_minstep_{min_step}{approach}/{file[:-5]}.json'
            dumping_path = f'results/precond_gen/{bench_type}/{goal}/{variables}/iter_{max_iter}_lr_{center_lr}_{scale_lr}_minstep_{min_step}{approach}/{file[:-5]}_data.pt'

            tot_success, tot_bugs, result_details, running_times, running_iters = precondition_gen(os.path.join(nowdir, file), goal, variables, debug=False,
                                                                                                   max_iter=max_iter, center_lr=center_lr, scale_lr=scale_lr, min_step=min_step, approach=approach,
                                                                                                   dumping_path=dumping_path)

            pkl_package = {'time_stat': running_times, 'iter_stat': running_iters, 'numerical_bugs': tot_bugs, 'success_cnt': tot_success,
                           'precond_stat': result_details}
            print(pkl_package)

            if not os.path.exists(os.path.dirname(pkl_path)):
                os.makedirs(os.path.dirname(pkl_path))
            with open(pkl_path, 'wb') as f:
                pickle.dump(pkl_package, f)
            with open(json_path, 'w') as f:
                json.dump(pkl_package, f, indent=2)
            print(f'saved to {pkl_path}')

            global_tot_bugs += tot_bugs
            global_tot_success += tot_success

            if tot_success > 0:
                print(tot_success)
                success_cases.append(file)
            else:
                failed_cases.append(file)

            for det in result_details.values():
                average_shrink.append(det['average_shrinkage'])
                print('now avg shrinkage:', det['average_shrinkage'])

    print(f'{global_tot_bugs} bugs, {global_tot_success} out of them succeed to secure')
    print('success cases:', success_cases)
    print('failed cases:', failed_cases)
    print('Unsupported Ops:', global_unsupported_ops)
    print('Tot Cases:', len(average_shrink))
    print('Mean Shrinkage:', np.mean(average_shrink))