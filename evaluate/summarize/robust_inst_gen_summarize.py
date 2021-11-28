"""
    Summarize the output of bug verifier
"""
import os
import pickle
import json
import math
import numpy as np
from evaluate.seeds import seeds

# should be grist and/or debar
benchmarks = ['grist']

# account for different tries
summary_no = 1

config = 'v1_lr1_step500_0.1_iter1000'

if __name__ == '__main__':
    for bench in benchmarks:
        print(f'Now on {bench}')
        nowdir = f'results/inference_inst_gen/{bench}/{config}'
        output_js_path = f'results/summary/inference_inst_gen_{bench}_{summary_no}.json'

        out = dict()

        files = [item for item in os.listdir(os.path.join(nowdir, str(seeds[0]), 'stats'))]
        files = sorted(files)
        for file in files:

            tot_success_lst = list()
            avg_gen_time_lst = list()
            relative_span_lst = list()
            changed_nodes_lst = list()
            average_drist_lst = list()
            iters_lst = list()

            for seed in seeds:
                now_seed_status = dict()
                now_seed_dir = f'{nowdir}/{seed}/stats/{file}'

                tot_err = len([ff for ff in os.listdir(now_seed_dir) if ff.endswith('.json')])

                tot_success = 0
                tot_gen_time = 0.
                tot_nodes = 0
                relative_span = 0.
                changed_nodes = 0.
                average_drift = 0.
                iters = 0.

                for now_err_json in [ff for ff in os.listdir(now_seed_dir) if ff.endswith('.json')]:
                    with open(os.path.join(now_seed_dir, now_err_json), 'r') as f:
                        data = json.load(f)
                    tot_success += int(data['success'])
                    tot_gen_time += data['gentime']
                    iters += data['iters']
                    if data['success']:
                        relative_span += data['details']['span_len']
                        changed_nodes += data['details']['tot_unchanged_nodes']
                        tot_nodes = data['details']['tot_nodes']
                        tmp = data['details']['average_relative_drift']
                        if not math.isnan(tmp):
                            average_drift += tmp

                if tot_success > 0:
                    relative_span /= float(tot_success)
                    changed_nodes /= tot_success
                    average_drift /= float(tot_success)
                avg_gen_time = tot_gen_time

                tot_success_lst.append(tot_success)
                avg_gen_time_lst.append(avg_gen_time)
                relative_span_lst.append(relative_span)
                changed_nodes_lst.append(changed_nodes)
                average_drist_lst.append(average_drift)
                iters_lst.append(iters)

            now_file_aggregate = {
                'tot_err': tot_err,
                'tot_success': (int(np.mean(tot_success_lst)), int(np.std(tot_success_lst, ddof=1))),
                'tot_success_rate': (np.mean(tot_success_lst) / tot_err, '%'),
                'gen_time': (np.mean(avg_gen_time_lst), np.std(avg_gen_time_lst, ddof=1)),
                'relative_span': np.mean(relative_span_lst) if tot_success_lst[-1] > 0 else '/',
                'tot_node': tot_nodes,
                'tot_change_node': (np.mean(changed_nodes_lst)) if tot_success_lst[-1] > 0 else '/',
                'tot_change_node_rate': ((np.mean(changed_nodes_lst) / tot_nodes) if tot_success_lst[-1] > 0 else '/', '%'),
                'relative_drift': (np.mean(average_drist_lst), np.std(average_drist_lst, ddof=1)) if tot_success_lst[-1] > 0 else '/',
                'iter': (int(np.mean(iters_lst)), int(np.std(iters_lst, ddof=1)))
            }

            out[file] = now_file_aggregate

            print(json.dumps(now_file_aggregate, indent=2))

        print('summarize to', output_js_path)
        with open(output_js_path, 'w') as f:
            json.dump(out, f)

