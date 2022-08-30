
# DEFAULT_LR = 1
# DEFAULT_LR_DECAY = 0.1
# DEFAULT_ITERS = 100
# DEFAULT_STEP = 70


DEFAULT_LR = 10
DEFAULT_LR_DECAY = 0.1
DEFAULT_ITERS = 10
DEFAULT_STEP = 70

# DEFAULT_SPAN = 1e-4
#
# customize_span = {
#     '48a': 1e-5,
#     '61': 1e-5
# }

skip_stages_by_name = {
    '17': ['spec-input-rand-weight']
}


# should be grist and/or debar
run_benchmarks = ['grist']

# ver_code = f'v1_lr{DEFAULT_LR}_step{DEFAULT_STEP}_{DEFAULT_LR_DECAY}_iter{DEFAULT_ITERS}'
# ver_code = f'v2_lr{DEFAULT_LR}_step{DEFAULT_STEP}_{DEFAULT_LR_DECAY}_iter{DEFAULT_ITERS}'

# ver_code = f'v3_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'
# v4 changes the loss formulation
# ver_code = f'v4_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'

# v5 uses correct matmul abstraction
ver_code = f'v5_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'

import os
import time
import json
import yaml
from evaluate.seeds import seeds

if __name__ == '__main__':

    ordering = os.listdir('model_zoo/grist_protobufs_onnx')
    ordering = [x[:-5] for x in ordering if x.endswith('.onnx')]
    ordering = sorted(ordering, key=lambda x: int(''.join([y for y in x if '0' <= y <= '9'])) +
                                              sum([0.01 * ord(y) - 0.01 * ord('a') for y in x if 'a' <= y <= 'z']))


    defect_node_lists = []
    tot_time = 0.

    tot_random_succeed = 0
    tot_random_time = 0.

    # add number of nodes and number of defect nodes
    for i, mname in enumerate(ordering):
        # detection results
        with open(f'results/endtoend/detection/{mname}.json', 'r') as f:
            data = json.load(f)
        print(mname, ':')

        success_cnt = {}
        category_cnt = {}
        inst_time = 0.
        for seed in seeds:
            stats_dir = f'results/inference_inst_gen/grist/{ver_code}/{seed}/stats'

            with open(os.path.join(stats_dir, mname, f'data.json'), 'r') as f:
                stats = json.load(f)
            for k in stats:
                if k not in success_cnt:
                    success_cnt[k] = 0
                success_cnt[k] += int(stats[k]['success'])
                if k not in category_cnt:
                    category_cnt[k] = {}
                if stats[k]['category'] not in category_cnt[k]:
                    category_cnt[k][stats[k]['category']] = 0
                category_cnt[k][stats[k]['category']] += 1

                tot_time += stats[k]['time']
                inst_time += stats[k]['time']
            # print(stats)

        print(success_cnt)
        print(category_cnt)
        print('  time on this inst:', inst_time)

        # random approach results
        for seed in seeds:
            stats_path = f'results/endtoend/unittest/random/{seed}/grist/{mname}.json'
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                print(stats)
                for k, v in stats.items():
                    tot_random_time += v['tot_time']
                    tot_random_succeed += v['success_num'] > 0

        defect_node_lists.append({'name': mname, 'nodes': list(success_cnt.keys())[:1]})

    # with open('model_zoo/grist_defect_node.yaml', 'w') as f:
    #     yaml.dump(defect_node_lists, f)
    print('tot time =', tot_time)
    print('')
    print('tot random time =', tot_random_time)
    print('tot random succeed =', tot_random_succeed)