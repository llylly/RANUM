"""
    This script evalautes the end-to-end failure-triggering ability
"""



DEFAULT_LR = 1
DEFAULT_LR_DECAY = 0.1
DEFAULT_ITERS = 100
DEFAULT_STEP = 70
# DEFAULT_SPAN = 1e-4


# should be grist and/or debar
run_benchmarks = ['grist']

# ver_code = f'v1_lr{DEFAULT_LR}_step{DEFAULT_STEP}_{DEFAULT_LR_DECAY}_iter{DEFAULT_ITERS}'
# ver_code = f'v2_lr{DEFAULT_LR}_step{DEFAULT_STEP}_{DEFAULT_LR_DECAY}_iter{DEFAULT_ITERS}'

# ver_code = f'v3_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'
# v4 changes the loss formulation
# ver_code = f'v4_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'

# v5 uses correct matmul abstraction
# ver_code = f'v5_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'


# ver_code = f''

# annotation = ''
# annotation = 'baseline/gradient_descent/'

'''
 random = ver_code '' + annotation ''
 gd = ver_code 'v5xxxx' + annotation 'baseline/gradient_gradient'
 debarus = ver_code 'v5xxxx' + annotation ''
'''

N = 1000
batch_size = 100
abst_gen_time_limit = 180

import argparse
import os
import time
import json
import numpy as np

import torch


from interp.interp_module import load_onnx_from_file
from evaluate.seeds import seeds
from interp.interp_operator import Abstraction
from interp.specified_vars import nospan_vars
from interp.specified_vars import nodiff_vars as nodiff_vars_default
from trigger.inference.robust_inducing_inst_gen import expand

# whitelist = ['1']
whitelist = []
blacklist = []

stime = time.time()
def prompt(msg):
    print(f'[{time.time() - stime:.3f}s] ' + msg)


def err_trigger(modelpath, barefilename, ver_code, seed, annotation, ref_ver_code=''):

    data_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/{annotation}data'
    stats_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/{annotation}stats'
    ref_stats_dir = f'results/inference_inst_gen/{benchmark}/{ref_ver_code}/{seed}/{annotation}stats'
    ref_data_dir = f'results/inference_inst_gen/{benchmark}/{ref_ver_code}/{seed}/{annotation}data'

    print('data_dir =', data_dir, 'stats_dir =', stats_dir)

    model = load_onnx_from_file(modelpath,
                                customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
    prompt(f'{barefilename} model initialized')

    bare_name = modelpath.split('/')[-1].split('.')[0]

    initial_errors = model.analyze(model.gen_abstraction_heuristics(os.path.split(modelpath)[-1].split('.')[0]), {'average_pool_mode': 'coarse', 'diff_order': 1})

    if len(initial_errors) == 0:
        print('No numerical bug')
    else:
        print(f'{len(initial_errors)} possible numerical bug(s)')
        for k, v in initial_errors.items():
            print(f'- On tensor {k} triggered by operator {v[1]}:')
            for item in v[0]:
                print(str(item))

    # store original input abstraction for clipping
    vanilla_lb_ub = dict()
    # for s, abst in model.initial_abstracts.items():
    #     vanilla_lb_ub[s] = (torch.min(abst.lb).item(), torch.max(abst.ub).item())
    # # print(vanilla_lb_ub)

    range_for_sampling = dict()

    # obtain the expanded initial abstracts
    for s, abst in model.initial_abstracts.items():
        expanded_lb = expand(abst.lb, abst.shape, abst.splits)
        expanded_ub = expand(abst.ub, abst.shape, abst.splits)
        vanilla_lb_ub[s] = (expanded_lb, expanded_ub)
        now_abst = Abstraction()
        now_abst.lb = expanded_lb
        now_abst.ub = expanded_ub
        now_abst.shape = abst.shape.copy()
        now_abst.splits = [list(range(n)) for n in abst.shape]
        now_abst.var_name = s
        range_for_sampling[s] = now_abst

    is_random_approach = True
    if ver_code != '':
        is_random_approach = False

    if not is_random_approach:
        with open(os.path.join(stats_dir, barefilename, 'data.json'), 'r') as f:
            gen_st_dict = json.load(f)
    else:
        with open(os.path.join(ref_stats_dir, barefilename, 'data.json'), 'r') as f:
            gen_st_dict = json.load(f)

    overall_stat = dict()

    gen_time = 0.
    infer_time = 0.

    for err_node in initial_errors:
        print(f'now on error node {err_node}')
        now_stats = dict()
        local_st = gen_st_dict[err_node]
        span_len = local_st.get('span_len', 0.)
        err_seq_no = local_st['err_seq_no']
        if not is_random_approach:
            if local_st['success'] is False or span_len <= 1e-10:
                now_stats['success_num'] = 0
                continue

            print(local_st)
            print(f'load the interval')

            diverse = False

            failure_interval_centers = torch.load(os.path.join(data_dir, barefilename, f'{err_seq_no}_{err_node}_gen.pt'))
            for s, abst in range_for_sampling.items():
                abst.lb = failure_interval_centers[s] - span_len * (vanilla_lb_ub[s][1] - vanilla_lb_ub[s][0])
                abst.ub = failure_interval_centers[s] + span_len * (vanilla_lb_ub[s][1] - vanilla_lb_ub[s][0])
                if torch.max(vanilla_lb_ub[s][1] - vanilla_lb_ub[s][0]) > 1e-10:
                    diverse = True
                abst.lb = torch.minimum(torch.maximum(abst.lb, vanilla_lb_ub[s][0]), vanilla_lb_ub[s][1])
                abst.ub = torch.minimum(torch.maximum(abst.ub, vanilla_lb_ub[s][0]), vanilla_lb_ub[s][1])

            if not diverse:
                now_stats['success_num'] = 0
                continue

        # start the generation process
        success_num = 0
        tot_num = 0
        t0 = time.time()
        while tot_num < N:
            data = dict()
            t1 = time.time()
            for s, abst in range_for_sampling.items():
                thing = torch.rand([batch_size] + range_for_sampling[s].shape).type(abst.lb.dtype)
                scaled_thing = abst.lb + thing * (abst.ub - abst.lb)
                data[s] = scaled_thing.detach()
            t2 = time.time()
            gen_time += t2 - t1
            if not is_random_approach:
                success_num += batch_size
                tot_num += batch_size
                continue
            batch_abst = list()
            for i in range(batch_size):
                sample = dict()
                for s, orig_abst in range_for_sampling.items():
                    a = Abstraction()
                    a.lb = data[s][i]
                    a.ub = data[s][i]
                    a.splits = orig_abst.splits
                    a.shape = orig_abst.shape
                    a.var_name = orig_abst.var_name
                    sample[s] = a
                batch_abst.append(sample)
            t1 = time.time()
            for i in range(batch_size):
                _, errors = model.forward(batch_abst[i], {'diff_order': 0, 'concrete_rand': seed, 'continue_prop': False})
                if infer_time + time.time() - t1 > abst_gen_time_limit:
                    # TLE
                    break
                if err_node in errors:
                    success_num += 1
                    if success_num == 1:
                        # first data, saved as random.pt
                        success_sample = dict()
                        for s in data:
                            success_sample[s] = data[s][i]
                        torch.save(success_sample, os.path.join(ref_data_dir, barefilename, f'{err_seq_no}_{err_node}_random.pt'))
                tot_num += 1
                print(f'    {barefilename} {err_node} {success_num} / {tot_num}')
            t2 = time.time()
            infer_time += t2 - t1
            if not is_random_approach:
                assert success_num == tot_num
            if time.time() - t0 > abst_gen_time_limit:
                # TLE
                break

        now_stats['success_num'] = success_num
        now_stats['tot_num'] = tot_num
        now_stats['infer_time'] = infer_time
        now_stats['gen_time'] = gen_time

        overall_stat[err_node] = now_stats

    return overall_stat

parser = argparse.ArgumentParser()
parser.add_argument('method', choices=['debarus', 'gd', 'random'])

if __name__ == '__main__':

    args = parser.parse_args()

    if args.method == 'debarus':
        ver_code = f'v5_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'
        annotation = ''
    elif args.method == 'gd':
        ver_code = f'v5_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'
        annotation = 'baseline/gradient_descent/'
    elif args.method == 'random':
        ver_code = ''
        annotation = ''
        ref_ver_code = f'v5_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'

    output_dir = 'results/endtoend/unittest/'
    if ver_code == '':
        output_dir += 'random'
    else:
        output_dir += ver_code + annotation

    for seed in seeds:
        print('*' * 20)
        print('*' * 20)
        print('seed =', seed)

        shorten_statuses = dict()

        for benchmark in run_benchmarks:

            if not os.path.exists(os.path.join(output_dir, str(seed), benchmark)):
                os.makedirs(os.path.join(output_dir, str(seed), benchmark))

            if benchmark == 'grist':
                print('on GRIST bench')
                nowdir = 'model_zoo/grist_protobufs_onnx'
            elif benchmark == 'debar':
                print('on DEBAR bench')
                nowdir = 'model_zoo/tf_protobufs_onnx'
            files = sorted([x for x in os.listdir(nowdir) if x.endswith('.onnx')])
            nowlen = len(files)

            statuses = dict()

            for id, file in enumerate(files):
                print(f'[{id+1} / {nowlen}] {file}')
                barefilename = file.rsplit('.', maxsplit=1)[0]
                if len(whitelist) > 0 and barefilename not in whitelist: continue
                if barefilename in blacklist: continue
                # if os.path.exists(os.path.join(output_dir, str(seed), benchmark, f'{barefilename}.json')): continue
                status_dict = err_trigger(os.path.join(nowdir, file), barefilename, ver_code, seed, annotation, ref_ver_code=ref_ver_code if args.method == 'random' else '')

                with open(os.path.join(output_dir, str(seed), benchmark, f'{barefilename}.json'), 'w') as f:
                    json.dump(status_dict, fp=f, indent=2)
                    print(json.dumps(status_dict, indent=2))
