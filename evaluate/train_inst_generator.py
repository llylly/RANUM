"""
    This script runs whole benchmark set only for the purpose of generating training input for numerical bugs that require specified weights to trigger.
    Thus, we can check whether the specified weights is training feasible or not.
    It records the detailed status (success or not, etc) and running time statistics.
"""


# should be grist and/or debar
run_benchmarks = ['grist']

from config import DEFAULT_LR, DEFAULT_LR_DECAY, DEFAULT_ITERS, DEFAULT_STEP

# v5: corresponding robsut inducing inst version
infer_inst_ver_code = f'v5_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'

learning_rate = 1.

import os
import time
import json
import argparse

import torch

from interp.interp_module import load_onnx_from_file
from interp.specified_vars import nodiff_vars, nospan_vars
from trigger.inference.robust_inducing_inst_gen import InducingInputGenModule, inducing_inference_inst_gen
from trigger.train.train_generator import train_input_gen
from trigger.hints import customized_lr_inference_inst_gen
from evaluate.seeds import seeds

whitelist = []
# blacklist = []
blacklist = []

# I found the random input random weights for GRIST ID 17 causes shape mismatch bug, which impedes the full pass to the final leaf loss node.
# Therefore, we need to skip ID 17

success_cases = list()
fail_cases = list()
mode_stats = dict()
time_stats = dict()
iter_stats = dict()

# DEBARUS
# approach = ''
# Gradient Descent
# approach = 'gd'
# Random
approach = 'random'

parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, choices=['debarus', 'random', 'random_p_debarus', 'debarus_p_random'], default='debarus')
if __name__ == '__main__':

    args = parser.parse_args()
    approach = args.method
    # if approach == 'debarus': approach = ''

    output_str = ''

    for infer_inst_seed in seeds:

        for benchmark in run_benchmarks:

            if benchmark == 'grist':
                print('on GRIST bench')
                modeldir = 'model_zoo/grist_protobufs_onnx'
            elif benchmark == 'debar':
                print('on DEBAR bench')
                modeldir = 'model_zoo/tf_protobufs_onnx'

            infer_inst_dir = f'results/inference_inst_gen/{benchmark}/{infer_inst_ver_code}/{infer_inst_seed}'
            train_inst_dir = f'results/training_inst_gen/{benchmark}/{infer_inst_ver_code}{approach if approach != "debarus" else ""}/{infer_inst_seed}'

            files = os.listdir(f'{infer_inst_dir}/stats')

            tot_worked_items = 0
            tot_success = 0

            for id, file in enumerate(files):
                if len(whitelist) > 0 and file not in whitelist: continue
                if file in blacklist: continue

                train_inst_stats = dict()

                tread_s = time.time()
                with open(f'{infer_inst_dir}/stats/{file}/data.json', 'r') as f:
                    stats = json.load(f)
                if approach == 'random' or approach == 'random_p_debarus':
                    with open(f'results/endtoend/unittest/random/{infer_inst_seed}/{benchmark}/{file}.json', 'r') as f:
                        random_stats = json.load(f)
                tread_t = time.time()
                for err_node in stats:
                    detail_status = stats[err_node]

                    if approach == 'random' or approach == 'random_p_debarus':
                        detail_random_status = random_stats[err_node]
                        if detail_status['success'] and detail_random_status['success_num'] > 0:
                            tot_worked_items += 1
                            print(f'Handling #{tot_worked_items} ({id}){file}/{err_node}')

                            err_seq_no = detail_status['err_seq_no']

                            module, success, mode, num_iters, spend_time = train_input_gen(f'{modeldir}/{file}.onnx', err_node, err_seq_no, f'{infer_inst_dir}/data', infer_inst_seed, learning_rate,
                                                                                           approach=approach)
                            tot_success += int(success)

                            print(f'###### success on {tot_success} out of {tot_worked_items}, status mode = {mode}')

                            train_inst_stats[err_node] = {
                                'success': success,
                                'detail': mode,
                                'iters': num_iters,
                                'time': spend_time
                            }

                            if success > 0:
                                success_cases.append((file, err_node))

                                data_dir = f'{train_inst_dir}/data/{file}'
                                if not os.path.exists(os.path.dirname(os.path.join(data_dir, f'{err_seq_no}_{err_node}_train.pt'))):
                                    os.makedirs(os.path.dirname(os.path.join(data_dir, f'{err_seq_no}_{err_node}_train.pt')))
                                if module is not None:
                                    torch.save(module.dump_gen_inputs(), os.path.join(data_dir, f'{err_seq_no}_{err_node}_train.pt'))
                            else:
                                fail_cases.append((file, err_node))
                        else:
                            print(f'not success on ({id}){file}/{err_node}: skip')

                            train_inst_stats[err_node] = {
                                'success': False,
                                'detail': 'failed on inference generation',
                            }

                    if approach == 'debarus' or approach == 'debarus_p_random':
                        if detail_status['success']:
                            now_category = detail_status['category']
                            if now_category == 'spec-input-spec-weight':
                                tot_worked_items += 1
                                print(f'Handling #{tot_worked_items} ({id}){file}/{err_node}')

                                err_seq_no = detail_status['err_seq_no']

                                module, success, mode, num_iters, spend_time = train_input_gen(f'{modeldir}/{file}.onnx', err_node, err_seq_no, f'{infer_inst_dir}/data', infer_inst_seed, learning_rate,
                                                                                               approach=approach)
                                tot_success += int(success)

                                print(f'###### success on {tot_success} out of {tot_worked_items}, status mode = {mode}')

                                train_inst_stats[err_node] = {
                                    'success': success,
                                    'detail': mode,
                                    'iters': num_iters,
                                    'time': spend_time
                                }

                                if success > 0:
                                    success_cases.append((file, err_node))

                                    data_dir = f'{train_inst_dir}/data/{file}'
                                    if not os.path.exists(os.path.dirname(os.path.join(data_dir, f'{err_seq_no}_{err_node}_train.pt'))):
                                        os.makedirs(os.path.dirname(os.path.join(data_dir, f'{err_seq_no}_{err_node}_train.pt')))
                                    if module is not None:
                                        torch.save(module.dump_gen_inputs(), os.path.join(data_dir, f'{err_seq_no}_{err_node}_train.pt'))
                                else:
                                    fail_cases.append((file, err_node))
                            else:
                                train_inst_stats[err_node] = {
                                    'success': True,
                                    'detail': now_category,
                                    'iters': 0,
                                    'time': tread_t - tread_s
                                }
                                success_cases.append((file, err_node))
                        else:
                            print(f'not success on ({id}){file}/{err_node}: skip')
                            train_inst_stats[err_node] = {
                                'success': False,
                                'detail': 'inference generation failed'
                            }
                            fail_cases.append((file, err_node))

                if not os.path.exists(f'{train_inst_dir}/stats/{file}'):
                    os.makedirs(f'{train_inst_dir}/stats/{file}')
                with open(f'{train_inst_dir}/stats/{file}/data.json', 'w') as f:
                    json.dump(train_inst_stats, f, indent=2)


        print('success cases:', success_cases)
        print('fail cases:   ', fail_cases)

        output_str += f'{infer_inst_seed} success cases: (tot = {len(success_cases)}) {success_cases}\n fail cases: (tot = {len(fail_cases)}) {fail_cases}\n\n'

    print(output_str)
