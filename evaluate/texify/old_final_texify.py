"""
    This script yields the final table presented in the paper
"""

import os
import json
import numpy as np

from evaluate.seeds import seeds

DEFAULT_LR = 1
DEFAULT_LR_DECAY = 0.1
DEFAULT_ITERS = 100
DEFAULT_STEP = 70
ver_code = f'v5_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'

max_iter = 1000
center_lr = 0.1
scale_lr = 0.1
min_step = 0.01

precond_code = f'iter_{max_iter}_lr_{center_lr}_{scale_lr}_minstep_{min_step}'

unit_test_gen_time_limit = 180


debar_supports = ['1', '2a', '2b', '3', '6', '7', '8', '9a', '9b', '10', '11a', '11b', '11c', '14', '15', '16a', '16b', '16c', '18', '19','20','21','24','25','28a','28b','28c','28d','29','30', '31', '35a', '36a', '39a', '43a', '44', '45a', '45b', '48a', '48b', '49a', '50', '52', '55', '58', '59', '60', '61']
debar_unknown = ['28e', '35c', '36c', '39c', '43c', '49c']

def tof(x):
    if x > 0: return '$\\checkmark$'
    else: return '$\\times$'


if __name__ == '__main__':

    ordering = os.listdir('model_zoo/grist_protobufs_onnx')
    ordering = [x[:-5] for x in ordering if x.endswith('.onnx')]
    ordering = sorted(ordering, key=lambda x: int(''.join([y for y in x if '0' <= y <= '9'])) +
                                              sum([0.01 * ord(y) - 0.01 * ord('a') for y in x if 'a' <= y <= 'z']))

    ans = [[x] + ['' for _ in range(13)] for x in ordering]

    sizes = list()

    # print(ordering)

    tot_defects = 0
    tot_debar_ub = 0
    tot_debar_lb = 0

    tot_unittest_debar = 0
    tot_unittest_debar_time = 0.
    # tot_unittest_debar_time_proj10 = 0.
    tot_unittest_gd = 0
    tot_unittest_gd_time = 0.
    # tot_unittest_gd_time_proj10 = 0.
    tot_unittest_random = 0
    tot_unittest_random_time = 0.
    # tot_unittest_random_time_proj10 = 0.

    tot_systest_debar = 0
    tot_systest_debar_time = 0.
    tot_systest_random = 0
    tot_systest_random_time = 0.
    tot_systest_debarus_p_random = 0
    tot_systest_debarus_p_random_time = 0.
    tot_systest_random_p_debarus = 0
    tot_systest_random_p_debarus_time = 0.

    tot_precond_debar_weight_input = 0
    tot_precond_debar_weight_input_time = 0.
    tot_precond_gd_weight_input = 0
    tot_precond_gd_weight_input_time = 0.
    tot_precond_expand_weight_input = 0
    tot_precond_expand_weight_input_time = 0.
    tot_precond_debar_weight = 0
    tot_precond_debar_weight_time = 0.
    tot_precond_gd_weight = 0
    tot_precond_gd_weight_time = 0.
    tot_precond_expand_weight = 0
    tot_precond_expand_weight_time = 0.
    tot_precond_debar_input = 0
    tot_precond_debar_input_time = 0.
    tot_precond_gd_input = 0
    tot_precond_gd_input_time = 0.
    tot_precond_expand_input = 0
    tot_precond_expand_input_time = 0.

    # add number of nodes and number of defect nodes
    for i, mname in enumerate(ordering):
        # detection results
        with open(f'results/endtoend/detection/{mname}.json', 'r') as f:
            data = json.load(f)
        ans[i][1] = str(data['nodes'])
        sizes.append(data['nodes'])
        ans[i][2] = str(data['numerical_bugs'])
        # print(data)
        tot_defects += data['numerical_bugs']
        ans[i][12] = '$\checkmark$'
        ans[i][13] = '$\checkmark$' if mname in debar_supports else '?' if mname in debar_unknown else '$\\times$'
        if mname in debar_supports:
            tot_debar_ub += 1
            tot_debar_lb += 1
        if mname in debar_unknown:
            tot_debar_ub += 1

        # # unit test generation results
        #
        # # # 1: DEBARUS
        # num_debarus = 0
        # time_debarus = 0.
        # # time_debarus_proj = 0.
        # time1_debarus = 0.
        # time2_debarus = 0.
        # debarus_inference_status = dict()
        # for seed in seeds:
        #     with open(f'results/inference_inst_gen/grist/{ver_code}/{seed}/stats/{mname}/data.json', 'r') as f:
        #         data = json.load(f)
        #     with open(f'results/endtoend/unittest/{ver_code}/{seed}/grist/{mname}.json', 'r') as f:
        #         data2 = json.load(f)
        #     for err_node in data:
        #         time1 = min(data[err_node]['time'], unit_test_gen_time_limit)
        #         if data[err_node]['success'] and data[err_node]['time'] <= unit_test_gen_time_limit:
        #             time2 = data2[err_node]['gen_time']
        #             if time1 + time2 <= unit_test_gen_time_limit:
        #                 num_debarus += data2[err_node]['success_num']
        #                 time_debarus += time1 + time2
        #                 # time_debarus_proj += time1 + time2 * 10.
        #                 time1_debarus += time1
        #                 time2_debarus += time2
        #         else:
        #             time_debarus += time1
        #             # time_debarus_proj += time1
        #             time1_debarus += time1
        #     debarus_inference_status[seed] = data
        # ans[i][5] = f'{num_debarus} ({time_debarus:.2f})'
        # tot_unittest_debar += num_debarus
        # tot_unittest_debar_time += time_debarus
        # # tot_unittest_debar_time_proj10 += time_debarus_proj
        #
        # # 2: Gradient Descent
        # num_gd = 0
        # time_gd = 0.
        # # time_gd_proj = 0.
        # time1_gd = 0.
        # time2_gd = 0.
        # for seed in seeds:
        #     with open(f'results/inference_inst_gen/grist/{ver_code}/{seed}/baseline/gradient_descent/stats/{mname}/data.json', 'r') as f:
        #         data = json.load(f)
        #     with open(f'results/endtoend/unittest/{ver_code}baseline/gradient_descent/{seed}/grist/{mname}.json', 'r') as f:
        #         data2 = json.load(f)
        #     for err_node in data:
        #         time1 = min(data[err_node]['time'], unit_test_gen_time_limit)
        #         if data[err_node]['success'] and data[err_node]['time'] <= unit_test_gen_time_limit:
        #             time2 = data2[err_node]['gen_time']
        #             if time1 + time2 <= unit_test_gen_time_limit:
        #                 num_gd += data2[err_node]['success_num']
        #                 time_gd += time1 + time2
        #                 # time_gd_proj += time1 + time2 * 10.
        #                 time1_gd += time1
        #                 time2_gd += time2
        #         else:
        #             time_gd += time1
        #             # time_gd_proj += time1
        #             time1_gd += time1
        # ans[i][6] = f'{num_gd} ({time_gd:.2f})'
        # tot_unittest_gd += num_gd
        # tot_unittest_gd_time += time_gd
        # # tot_unittest_gd_time_proj10 += time_gd_proj
        #
        # # # 3: random
        # num_random = 0
        # time_random = 0.
        # time1_random = 0.
        # time2_random = 0.
        # failed_random_inference_dict = dict()
        # for seed in seeds:
        #     with open(f'results/endtoend/unittest/random/{seed}/grist/{mname}.json', 'r') as f:
        #         data2 = json.load(f)
        #     failed_random_inference_dict[seed] = list()
        #     for err_node in data2:
        #         if data2[err_node]['infer_time'] + data2[err_node]['gen_time'] <= unit_test_gen_time_limit:
        #             time1 = data2[err_node]['infer_time']
        #             time2 = data2[err_node]['gen_time']
        #             if data2[err_node]['success_num'] == 0:
        #                 failed_random_inference_dict[seed].append(err_node)
        #             if time1 + time2 <= unit_test_gen_time_limit:
        #                 num_random += data2[err_node]['success_num']
        #                 time_random += time1 + time2
        #                 time1_random += time1
        #                 time2_random += time2
        # ans[i][7] = f'{num_random} ({time_random:.2f})'
        # tot_unittest_random += num_random
        # tot_unittest_random_time += time_random

        # # system test generation results
        #
        # # 1: debarus
        # num_debarus = 0
        # time_debarus = 0.
        # for seed in seeds:
        #     with open(f'results/training_inst_gen/grist/{ver_code}/{seed}/stats/{mname}/data.json', 'r') as f:
        #         data = json.load(f)
        #     for err_node, inference_stat in debarus_inference_status[seed].items():
        #         if not inference_stat['success']:
        #             continue
        #         if data[err_node]['success']:
        #             num_debarus += 1
        #             time_debarus += data[err_node]['time']
        # ans[i][3] = f'{num_debarus} ({time_debarus:.2f})'
        # tot_systest_debar += num_debarus
        # tot_systest_debar_time += time_debarus
        #
        # # 2: random
        # num_random = 0
        # time_random = 0.
        # for seed in seeds:
        #     with open(f'results/training_inst_gen/grist/{ver_code}random/{seed}/stats/{mname}/data.json', 'r') as f:
        #         data = json.load(f)
        #     for err_node, inference_stat in debarus_inference_status[seed].items():
        #         if data[err_node]['success']:
        #             num_random += 1
        #             time_random += data[err_node]['time']
        # ans[i][4] = f'{num_random} ({time_random:.2f})'
        # tot_systest_random += num_random
        # tot_systest_random_time += time_random
        #
        # # 3: random_p_debarus
        # num_random = 0
        # time_random = 0.
        # for seed in seeds:
        #     with open(f'results/training_inst_gen/grist/{ver_code}random_p_debarus/{seed}/stats/{mname}/data.json', 'r') as f:
        #         data = json.load(f)
        #     for err_node, inference_stat in debarus_inference_status[seed].items():
        #         if data[err_node]['success']:
        #             num_random += 1
        #             time_random += data[err_node]['time']
        # ans[i][4] = f'{num_random} ({time_random:.2f})'
        # tot_systest_random_p_debarus += num_random
        # tot_systest_random_p_debarus_time += time_random
        #
        # # 4: debarus_p_random
        # num_random = 0
        # time_random = 0.
        # for seed in seeds:
        #     with open(f'results/training_inst_gen/grist/{ver_code}debarus_p_random/{seed}/stats/{mname}/data.json', 'r') as f:
        #         data = json.load(f)
        #     for err_node, inference_stat in debarus_inference_status[seed].items():
        #         if data[err_node]['success']:
        #             num_random += 1
        #             time_random += data[err_node]['time']
        # ans[i][4] = f'{num_random} ({time_random:.2f})'
        # tot_systest_debarus_p_random += num_random
        # tot_systest_debarus_p_random_time += time_random


        # precondition generation
        # weight + input

        # 1: debarus
        num_debarus = 0
        time_debarus = 0.
        with open(f'results/precond_gen/grist/all/all/{precond_code}/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarus += data['success_cnt']
            time_debarus += data['time_stat']['all']
        ans[i][8] = f'{tof(num_debarus)} ({time_debarus:.2f})'
        tot_precond_debar_weight_input += num_debarus
        tot_precond_debar_weight_input_time += time_debarus

        # 2: gd
        num_gd = 0
        time_gd = 0.
        with open(f'results/precond_gen/grist/all/all/{precond_code}gd/{mname}.json', 'r') as f:
            data = json.load(f)
            num_gd += data['success_cnt']
            time_gd += data['time_stat']['all']
        ans[i][9] = f'{tof(num_gd)} ({time_gd:.2f})'
        tot_precond_gd_weight_input += num_gd
        tot_precond_gd_weight_input_time += time_gd

        # 3: debarusexpand
        num_debarusexpand = 0
        time_debarusexpand = 0.
        with open(f'results/precond_gen/grist/all/all/{precond_code}ranumexpand/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarusexpand += data['success_cnt']
            time_debarusexpand += data['time_stat']['all']
        # ans[i][9] = f'{tof(num_gd)} ({time_gd:.2f})'
        tot_precond_expand_weight_input += num_debarusexpand
        tot_precond_expand_weight_input_time += time_debarusexpand



        # weight

        # 1: debarus
        num_debarus = 0
        time_debarus = 0.
        with open(f'results/precond_gen/grist/all/weight/{precond_code}/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarus += data['success_cnt']
            time_debarus += data['time_stat']['all']
        ans[i][10] = f'{tof(num_debarus)} ({time_debarus:.2f})'
        tot_precond_debar_weight += num_debarus
        tot_precond_debar_weight_time += time_debarus

        # 2: gd
        num_gd = 0
        time_gd = 0.
        with open(f'results/precond_gen/grist/all/weight/{precond_code}gd/{mname}.json', 'r') as f:
            data = json.load(f)
            num_gd += data['success_cnt']
            time_gd += data['time_stat']['all']
        ans[i][11] = f'{tof(num_gd)} ({time_gd:.2f})'
        tot_precond_gd_weight += num_gd
        tot_precond_gd_weight_time += time_gd

        # 3: debarusexpand
        num_debarusexpand = 0
        time_debarusexpand = 0.
        with open(f'results/precond_gen/grist/all/weight/{precond_code}ranumexpand/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarusexpand += data['success_cnt']
            time_debarusexpand += data['time_stat']['all']
        # ans[i][9] = f'{tof(num_gd)} ({time_gd:.2f})'
        tot_precond_expand_weight += num_debarusexpand
        tot_precond_expand_weight_time += time_debarusexpand

        # input

        # 1: debarus
        num_debarus = 0
        time_debarus = 0.
        with open(f'results/precond_gen/grist/all/input/{precond_code}/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarus += data['success_cnt']
            time_debarus += data['time_stat']['all']
        # ans[i][10] = f'{tof(num_debarus)} ({time_debarus:.2f})'
        tot_precond_debar_input += num_debarus
        tot_precond_debar_input_time += time_debarus

        # 2: gd
        num_gd = 0
        time_gd = 0.
        with open(f'results/precond_gen/grist/all/input/{precond_code}gd/{mname}.json', 'r') as f:
            data = json.load(f)
            num_gd += data['success_cnt']
            time_gd += data['time_stat']['all']
        # ans[i][11] = f'{tof(num_gd)} ({time_gd:.2f})'
        tot_precond_gd_input += num_gd
        tot_precond_gd_input_time += time_gd

        # 3: debarusexpand
        num_debarusexpand = 0
        time_debarusexpand = 0.
        with open(f'results/precond_gen/grist/all/input/{precond_code}ranumexpand/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarusexpand += data['success_cnt']
            time_debarusexpand += data['time_stat']['all']
        # ans[i][9] = f'{tof(num_gd)} ({time_gd:.2f})'
        tot_precond_expand_input += num_debarusexpand
        tot_precond_expand_input_time += time_debarusexpand




    # for item in ans:
    #     print(' & '.join(item) + ' \\\\')

    print('tot defects:                   ', tot_defects)
    print('tot debar:                     ', tot_debar_lb, '-', tot_debar_ub)
    print('unittest: debarus              ', tot_unittest_debar, 'time', tot_unittest_debar_time)
    print('unittest: gd                   ', tot_unittest_gd, 'time', tot_unittest_gd_time)
    print('unittest: random               ', tot_unittest_random, 'time', tot_unittest_random_time)
    print('systest: debarus               ', tot_systest_debar, 'time', tot_systest_debar_time)
    print('systest: random                ', tot_systest_random, 'time', tot_systest_random_time)
    print('systest: debarus_p_random      ', tot_systest_debarus_p_random, 'time', tot_systest_debarus_p_random_time)
    print('systest: random_p_debarus      ', tot_systest_random_p_debarus, 'time', tot_systest_random_p_debarus_time)
    print('precond: debarus weight + input', tot_precond_debar_weight_input, 'time', tot_precond_debar_weight_input_time)
    print('precond: dexpand weight + input', tot_precond_expand_weight_input, 'time', tot_precond_expand_weight_input_time)
    print('precond: gd weight + input     ', tot_precond_gd_weight_input, 'time', tot_precond_gd_weight_input_time)
    print('precond: debarus weight        ', tot_precond_debar_weight, 'time', tot_precond_debar_weight_time)
    print('precond: dexpand weight        ', tot_precond_expand_weight, 'time', tot_precond_expand_weight_time)
    print('precond: gd weight             ', tot_precond_gd_weight, 'time', tot_precond_gd_weight_time)
    print('precond: debarus input         ', tot_precond_debar_input, 'time', tot_precond_debar_input_time)
    print('precond: dexpand input         ', tot_precond_expand_input, 'time', tot_precond_expand_input_time)
    print('precond: gd input              ', tot_precond_gd_input, 'time', tot_precond_gd_input_time)

    # print('=' * 20, 'systest generation', '=' * 20)
    # col = 5
    # width = 6
    #
    # systest_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(tot_defects / col)) + 3)]
    #
    # row_p = 0
    # col_p = 0
    # defect_ordering = dict()
    #
    # # first, we need inference data to determine the defect ordering
    # debarus_inference_status = dict()
    # for i, mname in enumerate(ordering):
    #     for seed in seeds:
    #         with open(f'results/inference_inst_gen/grist/{ver_code}/{seed}/stats/{mname}/data.json', 'r') as f:
    #             data = json.load(f)
    #         if seed not in defect_ordering: defect_ordering[seed] = dict()
    #         if seed not in debarus_inference_status: debarus_inference_status[seed] = dict()
    #         defect_ordering[seed][mname] = list(data.keys())
    #         debarus_inference_status[seed][mname] = data
    #
    # # then, print out systest
    # defect_no = 0
    # for i, mname in enumerate(ordering):
    #
    #     if len(defect_ordering[seeds[0]][mname]) <= 1:
    #         systest_ans[row_p][col_p * width] = f'{mname}'
    #     else:
    #         systest_ans[row_p][col_p * width] = f'\\multirow{{{len(defect_ordering[seeds[0]][mname])}}}{{*}}{{' + mname + '}'
    #
    #     for err_node in defect_ordering[seeds[0]][mname]:
    #
    #         defect_no += 1
    #         systest_ans[row_p][col_p * width + 1] = f'{defect_no}'
    #
    #         # 1: debarus
    #         num_debarus = 0
    #         time_debarus = 0.
    #         for seed in seeds:
    #             with open(f'results/training_inst_gen/grist/{ver_code}/{seed}/stats/{mname}/data.json', 'r') as f:
    #                 data = json.load(f)
    #             inference_stat = debarus_inference_status[seed][mname][err_node]
    #             if not inference_stat['success']:
    #                 continue
    #             if data[err_node]['success']:
    #                 num_debarus += 1
    #                 time_debarus += data[err_node]['time']
    #         systest_ans[row_p][col_p * width + 2] = f'{num_debarus}'
    #         systest_ans[row_p][col_p * width + 3] = f'{time_debarus / 10.:.2f}'
    #
    #         # 2: random
    #         num_random = 0
    #         time_random = 0.
    #         for seed in seeds:
    #             with open(f'results/training_inst_gen/grist/{ver_code}random/{seed}/stats/{mname}/data.json', 'r') as f:
    #                 data = json.load(f)
    #             inference_stat = debarus_inference_status[seed][mname][err_node]
    #             if data[err_node]['success']:
    #                 num_random += 1
    #                 time_random += data[err_node]['time']
    #         systest_ans[row_p][col_p * width + 4] = f'{num_random}'
    #         systest_ans[row_p][col_p * width + 5] = f'{time_random / 10.:.2f}'
    #
    #         row_p += 1
    #
    #     if row_p >= np.ceil((tot_defects + 1) / col):
    #         row_p = 0
    #         col_p += 1
    #
    #
    # systest_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    # systest_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_systest_debar}' + '}'
    # systest_ans[row_p][col_p * width + 3] = '\\textbf{' + f'{tot_systest_debar_time / 10.:.2f}' + '}'
    # systest_ans[row_p][col_p * width + 4] = '\\textbf{' + f'{tot_systest_random}' + '}'
    # systest_ans[row_p][col_p * width + 5] = '\\textbf{' + f'{tot_systest_random_time / 10.:.2f}' + '}'
    #
    # for i, item in enumerate(systest_ans):
    #     if i == row_p:
    #         print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
    #     print(' & '.join(item) + ' \\\\')

    # print('=' * 20, 'unit generation', '=' * 20)
    # col = 3
    # width = 8
    #
    # unittest_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(tot_defects / col)) + 3)]
    #
    # row_p = 0
    # col_p = 0
    #
    # defect_no = 0
    # for i, mname in enumerate(ordering):
    #
    #     if len(defect_ordering[seeds[0]][mname]) <= 1:
    #         unittest_ans[row_p][col_p * width] = f'{mname}'
    #     else:
    #         unittest_ans[row_p][col_p * width] = f'\\multirow{{{len(defect_ordering[seeds[0]][mname])}}}{{*}}{{' + mname + '}'
    #
    #     for err_node in defect_ordering[seeds[0]][mname]:
    #
    #         defect_no += 1
    #         unittest_ans[row_p][col_p * width + 1] = f'{defect_no}'
    #
    #         # unit test generation results
    #
    #         # # 1: DEBARUS
    #         num_debarus = 0
    #         time_debarus = 0.
    #         # time_debarus_proj = 0.
    #         time1_debarus = 0.
    #         time2_debarus = 0.
    #         for seed in seeds:
    #             with open(f'results/inference_inst_gen/grist/{ver_code}/{seed}/stats/{mname}/data.json', 'r') as f:
    #                 data = json.load(f)
    #             with open(f'results/endtoend/unittest/{ver_code}/{seed}/grist/{mname}.json', 'r') as f:
    #                 data2 = json.load(f)
    #             time1 = min(data[err_node]['time'], unit_test_gen_time_limit)
    #             if data[err_node]['success'] and data[err_node]['time'] <= unit_test_gen_time_limit:
    #                 time2 = data2[err_node]['gen_time']
    #                 if time1 + time2 <= unit_test_gen_time_limit:
    #                     num_debarus += data2[err_node]['success_num']
    #                     time_debarus += time1 + time2
    #                     time1_debarus += time1
    #                     time2_debarus += time2
    #             else:
    #                 time_debarus += time1
    #                 time1_debarus += time1
    #         unittest_ans[row_p][col_p * width + 2] = f'{num_debarus / 10.:.1f}'
    #         unittest_ans[row_p][col_p * width + 3] = f'{time_debarus / 10.:.2f}'
    #
    #         # 2: Gradient Descent
    #         num_gd = 0
    #         time_gd = 0.
    #         # time_gd_proj = 0.
    #         time1_gd = 0.
    #         time2_gd = 0.
    #         for seed in seeds:
    #             with open(f'results/inference_inst_gen/grist/{ver_code}/{seed}/baseline/gradient_descent/stats/{mname}/data.json', 'r') as f:
    #                 data = json.load(f)
    #             with open(f'results/endtoend/unittest/{ver_code}baseline/gradient_descent/{seed}/grist/{mname}.json', 'r') as f:
    #                 data2 = json.load(f)
    #             time1 = min(data[err_node]['time'], unit_test_gen_time_limit)
    #             if data[err_node]['success'] and data[err_node]['time'] <= unit_test_gen_time_limit:
    #                 time2 = data2[err_node]['gen_time']
    #                 if time1 + time2 <= unit_test_gen_time_limit:
    #                     num_gd += data2[err_node]['success_num']
    #                     time_gd += time1 + time2
    #                     time1_gd += time1
    #                     time2_gd += time2
    #             else:
    #                 time_gd += time1
    #                 time1_gd += time1
    #         unittest_ans[row_p][col_p * width + 4] = f'{num_gd / 10.:.1f}'
    #         unittest_ans[row_p][col_p * width + 5] = f'{time_gd / 10.:.2f}'
    #
    #         # # 3: random
    #         num_random = 0
    #         time_random = 0.
    #         time1_random = 0.
    #         time2_random = 0.
    #         failed_random_inference_dict = dict()
    #         for seed in seeds:
    #             with open(f'results/endtoend/unittest/random/{seed}/grist/{mname}.json', 'r') as f:
    #                 data2 = json.load(f)
    #             failed_random_inference_dict[seed] = list()
    #             if data2[err_node]['infer_time'] + data2[err_node]['gen_time'] <= unit_test_gen_time_limit:
    #                 time1 = data2[err_node]['infer_time']
    #                 time2 = data2[err_node]['gen_time']
    #                 if data2[err_node]['success_num'] == 0:
    #                     failed_random_inference_dict[seed].append(err_node)
    #                 if time1 + time2 <= unit_test_gen_time_limit:
    #                     num_random += data2[err_node]['success_num']
    #                     time_random += time1 + time2
    #                     time1_random += time1
    #                     time2_random += time2
    #         unittest_ans[row_p][col_p * width + 6] = f'{num_random / 10.:.1f}'
    #         unittest_ans[row_p][col_p * width + 7] = f'{time_random / 10.:.2f}'
    #
    #         row_p += 1
    #
    #     if row_p >= np.ceil((tot_defects + 1) / col):
    #         row_p = 0
    #         col_p += 1
    #
    # unittest_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    # unittest_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_unittest_debar / 10.:.1f}' + '}'
    # unittest_ans[row_p][col_p * width + 3] = '\\textbf{' + f'{tot_unittest_debar_time / 10.:.2f}' + '}'
    # unittest_ans[row_p][col_p * width + 4] = '\\textbf{' + f'{tot_unittest_gd / 10.:.1f}' + '}'
    # unittest_ans[row_p][col_p * width + 5] = '\\textbf{' + f'{tot_unittest_gd_time / 10.:.2f}' + '}'
    # unittest_ans[row_p][col_p * width + 6] = '\\textbf{' + f'{tot_unittest_random / 10.:.1f}' + '}'
    # unittest_ans[row_p][col_p * width + 7] = '\\textbf{' + f'{tot_unittest_random_time / 10.:.2f}' + '}'
    #
    # for i, item in enumerate(unittest_ans):
    #     if i == row_p:
    #         print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
    #     print(' & '.join(item) + ' \\\\')


    print('=' * 20, 'precondition generation weight + input', '=' * 20)
    col = 3
    width = 7

    precond_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(len(ordering) / col)) + 3)]

    row_p = 0
    col_p = 0

    for i, mname in enumerate(ordering):

        precond_ans[row_p][col_p * width] = f'{mname}'

        # precondition generation
        # weight + input

        # 1: debarus
        num_debarus = 0
        time_debarus = 0.
        with open(f'results/precond_gen/grist/all/all/{precond_code}/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarus += data['success_cnt']
            time_debarus += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'

        # 2: debarusexpand
        num_expand = 0
        time_expand = 0.
        with open(f'results/precond_gen/grist/all/all/{precond_code}ranumexpand/{mname}.json', 'r') as f:
            data = json.load(f)
            num_expand += data['success_cnt']
            time_expand += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'

        # 3: gd
        num_gd = 0
        time_gd = 0.
        with open(f'results/precond_gen/grist/all/all/{precond_code}gd/{mname}.json', 'r') as f:
            data = json.load(f)
            num_gd += data['success_cnt']
            time_gd += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'


        # # weight
        #
        # # 1: debarus
        # num_debarus = 0
        # time_debarus = 0.
        # with open(f'results/precond_gen/grist/all/weight/{precond_code}/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_debarus += data['success_cnt']
        #     time_debarus += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        # precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'
        #
        # # 2: debarusexpand
        # num_expand = 0
        # time_expand = 0.
        # with open(f'results/precond_gen/grist/all/weight/{precond_code}ranumexpand/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_expand += data['success_cnt']
        #     time_expand += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        # precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'
        #
        # # 3: gd
        # num_gd = 0
        # time_gd = 0.
        # with open(f'results/precond_gen/grist/all/weight/{precond_code}gd/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_gd += data['success_cnt']
        #     time_gd += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        # precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'
        #
        # # input
        #
        # # 1: debarus
        # num_debarus = 0
        # time_debarus = 0.
        # with open(f'results/precond_gen/grist/all/input/{precond_code}/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_debarus += data['success_cnt']
        #     time_debarus += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        # precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'
        #
        # # 2: debarusexpand
        # num_expand = 0
        # time_expand = 0.
        # with open(f'results/precond_gen/grist/all/input/{precond_code}ranumexpand/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_expand += data['success_cnt']
        #     time_expand += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        # precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'
        #
        # # 3: gd
        # num_gd = 0
        # time_gd = 0.
        # with open(f'results/precond_gen/grist/all/input/{precond_code}gd/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_gd += data['success_cnt']
        #     time_gd += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        # precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'

        row_p += 1

        if row_p >= np.ceil(len(ordering) / col):
            row_p = 0
            col_p += 1

    precond_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    precond_ans[row_p][col_p * width + 1] = '\\textbf{' + f'{tot_precond_debar_weight_input}' + '}'
    precond_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_precond_debar_weight_input_time:.2f}' + '}'
    precond_ans[row_p][col_p * width + 3] = '\\textbf{' + f'{tot_precond_expand_weight_input}' + '}'
    precond_ans[row_p][col_p * width + 4] = '\\textbf{' + f'{tot_precond_expand_weight_input_time:.2f}' + '}'
    precond_ans[row_p][col_p * width + 5] = '\\textbf{' + f'{tot_precond_gd_weight_input}' + '}'
    precond_ans[row_p][col_p * width + 6] = '\\textbf{' + f'{tot_precond_gd_weight_input_time:.2f}' + '}'

    for i, item in enumerate(precond_ans):
        if i == row_p:
            print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
        print(' & '.join(item) + ' \\\\')



    print('=' * 20, 'precondition generation weight', '=' * 20)
    col = 3
    width = 7

    precond_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(len(ordering) / col)) + 3)]

    row_p = 0
    col_p = 0

    for i, mname in enumerate(ordering):

        precond_ans[row_p][col_p * width] = f'{mname}'

        # # precondition generation
        # # weight + input
        #
        # # 1: debarus
        # num_debarus = 0
        # time_debarus = 0.
        # with open(f'results/precond_gen/grist/all/all/{precond_code}/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_debarus += data['success_cnt']
        #     time_debarus += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        # precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'
        #
        # # 2: debarusexpand
        # num_expand = 0
        # time_expand = 0.
        # with open(f'results/precond_gen/grist/all/all/{precond_code}ranumexpand/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_expand += data['success_cnt']
        #     time_expand += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        # precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'
        #
        # # 3: gd
        # num_gd = 0
        # time_gd = 0.
        # with open(f'results/precond_gen/grist/all/all/{precond_code}gd/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_gd += data['success_cnt']
        #     time_gd += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        # precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'


        # weight

        # 1: debarus
        num_debarus = 0
        time_debarus = 0.
        with open(f'results/precond_gen/grist/all/weight/{precond_code}/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarus += data['success_cnt']
            time_debarus += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'

        # 2: debarusexpand
        num_expand = 0
        time_expand = 0.
        with open(f'results/precond_gen/grist/all/weight/{precond_code}ranumexpand/{mname}.json', 'r') as f:
            data = json.load(f)
            num_expand += data['success_cnt']
            time_expand += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'

        # 3: gd
        num_gd = 0
        time_gd = 0.
        with open(f'results/precond_gen/grist/all/weight/{precond_code}gd/{mname}.json', 'r') as f:
            data = json.load(f)
            num_gd += data['success_cnt']
            time_gd += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'

        # # input
        #
        # # 1: debarus
        # num_debarus = 0
        # time_debarus = 0.
        # with open(f'results/precond_gen/grist/all/input/{precond_code}/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_debarus += data['success_cnt']
        #     time_debarus += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        # precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'
        #
        # # 2: debarusexpand
        # num_expand = 0
        # time_expand = 0.
        # with open(f'results/precond_gen/grist/all/input/{precond_code}ranumexpand/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_expand += data['success_cnt']
        #     time_expand += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        # precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'
        #
        # # 3: gd
        # num_gd = 0
        # time_gd = 0.
        # with open(f'results/precond_gen/grist/all/input/{precond_code}gd/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_gd += data['success_cnt']
        #     time_gd += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        # precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'

        row_p += 1

        if row_p >= np.ceil(len(ordering) / col):
            row_p = 0
            col_p += 1

    precond_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    precond_ans[row_p][col_p * width + 1] = '\\textbf{' + f'{tot_precond_debar_weight}' + '}'
    precond_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_precond_debar_weight_time:.2f}' + '}'
    precond_ans[row_p][col_p * width + 3] = '\\textbf{' + f'{tot_precond_expand_weight}' + '}'
    precond_ans[row_p][col_p * width + 4] = '\\textbf{' + f'{tot_precond_expand_weight_time:.2f}' + '}'
    precond_ans[row_p][col_p * width + 5] = '\\textbf{' + f'{tot_precond_gd_weight}' + '}'
    precond_ans[row_p][col_p * width + 6] = '\\textbf{' + f'{tot_precond_gd_weight_time:.2f}' + '}'

    for i, item in enumerate(precond_ans):
        if i == row_p:
            print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
        print(' & '.join(item) + ' \\\\')



    print('=' * 20, 'precondition generation input', '=' * 20)
    col = 3
    width = 7

    precond_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(len(ordering) / col)) + 3)]

    row_p = 0
    col_p = 0

    for i, mname in enumerate(ordering):

        precond_ans[row_p][col_p * width] = f'{mname}'

        # # precondition generation
        # # weight + input
        #
        # # 1: debarus
        # num_debarus = 0
        # time_debarus = 0.
        # with open(f'results/precond_gen/grist/all/all/{precond_code}/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_debarus += data['success_cnt']
        #     time_debarus += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        # precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'
        #
        # # 2: debarusexpand
        # num_expand = 0
        # time_expand = 0.
        # with open(f'results/precond_gen/grist/all/all/{precond_code}ranumexpand/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_expand += data['success_cnt']
        #     time_expand += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        # precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'
        #
        # # 3: gd
        # num_gd = 0
        # time_gd = 0.
        # with open(f'results/precond_gen/grist/all/all/{precond_code}gd/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_gd += data['success_cnt']
        #     time_gd += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        # precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'
        #
        #
        # # weight
        #
        # # 1: debarus
        # num_debarus = 0
        # time_debarus = 0.
        # with open(f'results/precond_gen/grist/all/weight/{precond_code}/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_debarus += data['success_cnt']
        #     time_debarus += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        # precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'
        #
        # # 2: debarusexpand
        # num_expand = 0
        # time_expand = 0.
        # with open(f'results/precond_gen/grist/all/weight/{precond_code}ranumexpand/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_expand += data['success_cnt']
        #     time_expand += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        # precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'
        #
        # # 3: gd
        # num_gd = 0
        # time_gd = 0.
        # with open(f'results/precond_gen/grist/all/weight/{precond_code}gd/{mname}.json', 'r') as f:
        #     data = json.load(f)
        #     num_gd += data['success_cnt']
        #     time_gd += data['time_stat']['all']
        # precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        # precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'

        # input

        # 1: debarus
        num_debarus = 0
        time_debarus = 0.
        with open(f'results/precond_gen/grist/all/input/{precond_code}/{mname}.json', 'r') as f:
            data = json.load(f)
            num_debarus += data['success_cnt']
            time_debarus += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 1] = f'{tof(num_debarus)}'
        precond_ans[row_p][col_p * width + 2] = f'{time_debarus:.2f}'

        # 2: debarusexpand
        num_expand = 0
        time_expand = 0.
        with open(f'results/precond_gen/grist/all/input/{precond_code}ranumexpand/{mname}.json', 'r') as f:
            data = json.load(f)
            num_expand += data['success_cnt']
            time_expand += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 3] = f'{tof(num_expand)}'
        precond_ans[row_p][col_p * width + 4] = f'{time_expand:.2f}'

        # 3: gd
        num_gd = 0
        time_gd = 0.
        with open(f'results/precond_gen/grist/all/input/{precond_code}gd/{mname}.json', 'r') as f:
            data = json.load(f)
            num_gd += data['success_cnt']
            time_gd += data['time_stat']['all']
        precond_ans[row_p][col_p * width + 5] = f'{tof(num_gd)}'
        precond_ans[row_p][col_p * width + 6] = f'{time_gd:.2f}'

        row_p += 1

        if row_p >= np.ceil(len(ordering) / col):
            row_p = 0
            col_p += 1

    precond_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    precond_ans[row_p][col_p * width + 1] = '\\textbf{' + f'{tot_precond_debar_input}' + '}'
    precond_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_precond_debar_input_time:.2f}' + '}'
    precond_ans[row_p][col_p * width + 3] = '\\textbf{' + f'{tot_precond_expand_input}' + '}'
    precond_ans[row_p][col_p * width + 4] = '\\textbf{' + f'{tot_precond_expand_input_time:.2f}' + '}'
    precond_ans[row_p][col_p * width + 5] = '\\textbf{' + f'{tot_precond_gd_input}' + '}'
    precond_ans[row_p][col_p * width + 6] = '\\textbf{' + f'{tot_precond_gd_input_time:.2f}' + '}'

    for i, item in enumerate(precond_ans):
        if i == row_p:
            print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
        print(' & '.join(item) + ' \\\\')

    print('=' * 20, 'detection', '=' * 20)
    col = 5
    width = 3

    det_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(len(ordering) / col)) + 3)]

    row_p = 0
    col_p = 0

    for i, mname in enumerate(ordering):

        det_ans[row_p][col_p * width] = f'{mname}'

        with open(f'results/endtoend/detection/{mname}.json', 'r') as f:
            data = json.load(f)
        det_ans[row_p][col_p * width + 1] = '$\checkmark$'
        det_ans[row_p][col_p * width + 2] = '$\checkmark$' if mname in debar_supports else '?' if mname in debar_unknown else '$\\times$'
        row_p += 1

        if row_p >= np.ceil((len(ordering) + 1) / col):
            row_p = 0
            col_p += 1

    det_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    det_ans[row_p][col_p * width + 1] = '\\textbf{' + f'{len(ordering)}' + '}'
    det_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_debar_lb}' + '}'

    for i, item in enumerate(det_ans):
        if i == row_p:
            print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
        print(' & '.join(item) + ' \\\\')

    print('=' * 20, 'size stats', '=' * 20)

    sizes = np.array(sizes)
    print('max size =', np.max(sizes))
    print('min size =', np.min(sizes))
    print('avg size =', np.mean(sizes))
    print('median size =', np.median(sizes))