"""
    This script runs whole benchmark set only for the purpose of generating training input for numerical bugs that require specified weights to trigger.
    Thus, we can check whether the specified weights is training feasible or not.
    It records the detailed status (success or not, etc) and running time statistics.
"""


# should be grist and/or debar
run_benchmarks = ['grist']


infer_inst_ver_code = 'v3_lr1_decay_0.1_step70_iter100'
infer_inst_seed = 188
learning_rate = 1.

import os
import time
import json

import torch

from interp.interp_module import load_onnx_from_file
from interp.specified_vars import nodiff_vars, nospan_vars
from trigger.inference.robust_inducing_inst_gen import InducingInputGenModule, inducing_inference_inst_gen
from trigger.train.train_generator import train_input_gen
from trigger.hints import customized_lr_inference_inst_gen
from evaluate.seeds import seeds

# whitelist = ['17']
whitelist = []
# blacklist = []
blacklist = ['17']

# I found the random input random weights for GRIST ID 17 causes shape mismatch bug, which impedes the full pass to the final leaf loss node.
# Therefore, we need to skip ID 17

success_cases = list()
fail_cases = list()
mode_stats = dict()
time_stats = dict()
iter_stats = dict()

if __name__ == '__main__':

    for benchmark in run_benchmarks:

        if benchmark == 'grist':
            print('on GRIST bench')
            modeldir = 'model_zoo/grist_protobufs_onnx'
        elif benchmark == 'debar':
            print('on DEBAR bench')
            modeldir = 'model_zoo/tf_protobufs_onnx'

        infer_inst_dir = f'results/inference_inst_gen/{benchmark}/{infer_inst_ver_code}/{infer_inst_seed}'
        train_inst_dir = f'results/training_inst_gen/{benchmark}/{infer_inst_ver_code}/{infer_inst_seed}'

        files = os.listdir(f'{infer_inst_dir}/stats')

        tot_worked_items = 0
        tot_success = 0

        for id, file in enumerate(files):
            if len(whitelist) > 0 and file not in whitelist: continue
            if file in blacklist: continue

            train_inst_stats = dict()

            with open(f'{infer_inst_dir}/stats/{file}/data.json', 'r') as f:
                stats = json.load(f)
            for err_node in stats:
                detail_status = stats[err_node]
                if detail_status['success']:
                    now_category = detail_status['category']
                    if now_category == 'spec-input-spec-weight':
                        tot_worked_items += 1
                        print(f'Handling #{tot_worked_items} ({id}){file}/{err_node}')

                        err_seq_no = detail_status['err_seq_no']

                        module, success, mode, num_iters, spend_time = train_input_gen(f'{modeldir}/{file}.onnx', err_node, err_seq_no, f'{infer_inst_dir}/data', infer_inst_seed, learning_rate)
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
                            torch.save(module.dump_gen_inputs(), os.path.join(data_dir, f'{err_seq_no}_{err_node}_train.pt'))
                        else:
                            fail_cases.append((file, err_node))

                else:
                    print(f'not success on ({id}){file}/{err_node}: skip')

            if not os.path.exists(f'{train_inst_dir}/stats/{file}'):
                os.makedirs(f'{train_inst_dir}/stats/{file}')
            with open(f'{train_inst_dir}/stats/{file}/data.json', 'w') as f:
                json.dump(train_inst_stats, f, indent=2)


    print('success cases:', success_cases)
    print('fail cases:   ', fail_cases)

