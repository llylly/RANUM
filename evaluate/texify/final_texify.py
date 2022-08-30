"""
    This script yields the final table presented in the paper
"""

import os
import json
import numpy as np

from dateutil import parser
from evaluate.seeds import seeds

from config import DEFAULT_LR, DEFAULT_LR_DECAY, DEFAULT_ITERS, DEFAULT_STEP
from evaluate.precond_generator import max_iter, center_lr, scale_lr, min_step

ver_code = f'v5_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'
precond_code = f'iter_{max_iter}_lr_{center_lr}_{scale_lr}_minstep_{min_step}'

# max_iter = 1000
# center_lr = 0.1
# scale_lr = 0.1
# min_step = 0.01


unit_test_gen_time_limit = 180


debar_supports = ['1', '2a', '2b', '3', '6', '7', '8', '9a', '9b', '10',
                  '11a', '11b', '11c', '14', '15', '16a', '16b', '16c', '18', '19', '20',
                  '21', '24', '25','28a', '28b', '28c', '28d', '29', '30',
                  '31', '35a', '36a', '39a',
                  '43a', '44', '45a', '45b', '48a', '48b', '49a', '50',
                  '52', '55', '58', '59', '60', '61']

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

    # =============== Defect Detection Table =================
    print('=' * 20, 'detection', '=' * 20)
    tot_ranum = 0
    tot_debar = 0
    tot_detect_time = 0.

    col = 5
    width = 3

    det_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(len(ordering) / col)))]

    row_p = 0
    col_p = 0

    for i, mname in enumerate(ordering):

        det_ans[row_p][col_p * width] = f'{mname}'

        with open(f'results/endtoend/detection/{mname}.json', 'r') as f:
            data = json.load(f)
        tot_detect_time += data['time_stat']['all']
        det_ans[row_p][col_p * width + 1] = '$\checkmark$' if data['numerical_bugs'] > 0 else '$\\times$'
        tot_ranum += bool(data['numerical_bugs'] > 0)
        det_ans[row_p][col_p * width + 2] = '$\checkmark$' if mname in debar_supports else '$\\times$'
        tot_debar += bool(mname in debar_supports)
        row_p += 1

        if row_p >= np.ceil((len(ordering) + 1) / col):
            row_p = 0
            col_p += 1

    det_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    det_ans[row_p][col_p * width + 1] = '\\textbf{' + f'{tot_ranum}' + '}'
    det_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_debar}' + '}'

    for i, item in enumerate(det_ans):
        if i == row_p:
            print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
        print(' & '.join(item) + ' \\\\')

    print('RANUM detection time:', tot_detect_time, 'avg time =', tot_detect_time / tot_ranum)

    # =============== Failure-Exhibiting Unit Test Table =================
    print('=' * 20, 'Failure-Exhibiting Unit Test', '=' * 20)

    "Read running time and success statistics of RANUM"
    grist_unit_test = {}
    for i, mname in enumerate(ordering):
        grist_unit_test[mname] = []
        if '0' <= mname[-1] <= '9':
            attach_name = '0'
            logmname = mname
        else:
            attach_name = str(ord(mname[-1]) - ord('a'))
            logmname = mname[:-1]
        for run in range(3):
            find = False
            with open(f'../GRIST/log_{logmname}_{attach_name}_run{run}.txt', 'r') as f:
                lines = f.readlines()
            for line in lines[-5:]:
                if line.startswith('FINAL RESULTS:'):
                    find = True
                    time_cost = line[line.find('Time cost:') + 10:]
                    time_cost = time_cost[time_cost.find('<')+1 :time_cost.find('>')]
                    time_obj = parser.parse(time_cost)
                    time_cost = time_obj.hour * 3600. + time_obj.minute * 60. + time_obj.second + time_obj.microsecond / 1000000.
                    success = line.count('Success') > 0
                    grist_unit_test[mname].append([success, time_cost])
                    break
            assert find or mname == '14'
    for i, mname in enumerate(ordering):
        run_records = grist_unit_test[mname]
        if len(run_records) > 2:
            grist_unit_test[mname] = [run_records[0][0] + run_records[1][0] + run_records[2][0],
                                      (run_records[0][1] + run_records[1][1] + run_records[2][1]) / 3.]
        else:
            assert mname == '14' # failed to run case ID using GRIST original code, so we directly use the running statistics from the GRIST paper
            grist_unit_test[mname] = [3, 86.23]
    grist_unit_test_fails = {
        '17': 0, '31': 3, '51': 3
    } # digested from GRIST paper since our reproduced failures are strictly more, and GRIST paper's number may be the better run
    grist_unit_test_gt_runtime = {
        '5': 0.34, '16c': 4.43, '28b': 176.02, '28d': 176.02, '49b': 307.24
        # digested these running time from GRIST paper because our reproduced running time is much longer than GRIST reported ones
    }
    for mname, dataitem in grist_unit_test.items():
        if mname not in grist_unit_test_fails:
            dataitem[0] = 10
        else:
            dataitem[0] = grist_unit_test_fails[mname]
        if mname in grist_unit_test_gt_runtime:
            dataitem[1] = grist_unit_test_gt_runtime[mname]

    ranum_unit_test = {}
    ranum_inference_status = dict([(seed, {}) for seed in seeds])
    for i, mname in enumerate(ordering):
        ranum_unit_test[mname] = [0, 0.]
        for seed in seeds:
            with open(f'results/inference_inst_gen/grist/{ver_code}/{seed}/stats/{mname}/data.json', 'r') as f:
                data = json.load(f)
                ranum_inference_status[seed][mname] = data
                assert len(data) == 1
                for k, v in data.items():
                    ranum_unit_test[mname][0] += int(v['success'])
                    ranum_unit_test[mname][1] += v['time']
        ranum_unit_test[mname][1] /= ranum_unit_test[mname][0]


    tot_ranum_unit_test = 0
    tot_grist_unit_test = 0
    tot_ranum_unit_test_time = 0.
    tot_grist_unit_test_time = 0.

    col = 4
    width = 6

    unit_test_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(len(ordering) / col)))]

    row_p = 0
    col_p = 0

    for i, mname in enumerate(ordering):

        unit_test_ans[row_p][col_p * width] = f'{mname}'

        unit_test_ans[row_p][col_p * width + 1] = f'{ranum_unit_test[mname][0]}'
        unit_test_ans[row_p][col_p * width + 2] = f'{ranum_unit_test[mname][1]:.2f}'
        if grist_unit_test[mname][0] == 0:
            unit_test_ans[row_p][col_p * width + 3] = f'$+\\infty$'
        elif ranum_unit_test[mname][1] <= grist_unit_test[mname][1]:
            unit_test_ans[row_p][col_p * width + 3] = f'{grist_unit_test[mname][1] / ranum_unit_test[mname][1]:.2f} X'
        else:
            unit_test_ans[row_p][col_p * width + 3] = f'{- ranum_unit_test[mname][1] / grist_unit_test[mname][1]:.2f} X'
        unit_test_ans[row_p][col_p * width + 4] = f'{grist_unit_test[mname][0]}'
        unit_test_ans[row_p][col_p * width + 5] = f'{grist_unit_test[mname][1]:.2f}' if grist_unit_test[mname][0] > 0 else '-'

        tot_ranum_unit_test += ranum_unit_test[mname][0]
        tot_ranum_unit_test_time += ranum_unit_test[mname][1]
        tot_grist_unit_test += grist_unit_test[mname][0]
        tot_grist_unit_test_time += grist_unit_test[mname][1] if grist_unit_test[mname][0] > 0 else 0.

        row_p += 1
        if row_p >= np.ceil((len(ordering) + 1) / col):
            row_p = 0
            col_p += 1

    unit_test_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    unit_test_ans[row_p][col_p * width + 1] = '\\textbf{' + f'{tot_ranum_unit_test}' + '}'
    unit_test_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_ranum_unit_test_time / len(ordering):.2f}' + '}'
    unit_test_ans[row_p][col_p * width + 3] = '\\textbf{' + f'{tot_grist_unit_test_time / tot_ranum_unit_test_time:.2f} X' + '}'
    unit_test_ans[row_p][col_p * width + 4] = '\\textbf{' + f'{tot_grist_unit_test}' + '}'
    unit_test_ans[row_p][col_p * width + 5] = '\\textbf{' + f'{tot_grist_unit_test_time / len(ordering):.2f}' + '}'

    for i, item in enumerate(unit_test_ans):
        if i == row_p:
            print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
        print(' & '.join(item) + ' \\\\')


    # =============== Failure-Exhibiting System Test Table =================
    print('=' * 20, 'Failure-Exhibiting System Test', '=' * 20)

    random_inference_status = dict([(seed, {}) for seed in seeds])
    for i, mname in enumerate(ordering):
        ranum_unit_test[mname] = [0, 0.]
        for seed in seeds:
            with open(f'results/endtoend/unittest/random/{seed}/grist/{mname}.json', 'r') as f:
                data = json.load(f)
                random_inference_status[seed][mname] = data
                assert len(data) == 1

    tot_systest_ranum = 0
    tot_systest_random = 0
    tot_systest_ranum_time = 0.
    tot_systest_random_time = 0.

    tot_systest_ranum_p_random = 0
    tot_systest_ranum_p_random_time = 0.
    tot_systest_random_p_ranum = 0
    tot_systest_random_p_ranum_time = 0.

    col = 5
    width = 5

    systest_ans = [['' for _ in range(col * width)] for _ in range(int(np.ceil(len(ordering) / col)))]

    row_p = 0
    col_p = 0

    # then, print out systest
    for i, mname in enumerate(ordering):

        systest_ans[row_p][col_p * width] = f'{mname}'

        err_node = list(ranum_inference_status[seed][mname].keys())[0]

        # 1: debarus
        num_debarus = 0
        time_debarus = 0.
        for seed in seeds:
            with open(f'results/training_inst_gen/grist/{ver_code}/{seed}/stats/{mname}/data.json', 'r') as f:
                data = json.load(f)
            inference_stat = ranum_inference_status[seed][mname][err_node]
            time_debarus += inference_stat['time']
            if not inference_stat['success']:
                continue
            if data[err_node]['success']:
                num_debarus += 1
                time_debarus += data[err_node]['time']
        systest_ans[row_p][col_p * width + 1] = f'{num_debarus}'
        systest_ans[row_p][col_p * width + 2] = f'{time_debarus / len(seeds):.2f}'

        # 2: random
        num_random = 0
        time_random = 0.
        for seed in seeds:
            with open(f'results/training_inst_gen/grist/{ver_code}random/{seed}/stats/{mname}/data.json', 'r') as f:
                data = json.load(f)
            inference_stat = random_inference_status[seed][mname][err_node]
            time_random += inference_stat['tot_time']
            if data[err_node]['success']:
                num_random += 1
                time_random += data[err_node]['time']
            else:
                time_random += 1800. # time limit is 1800, actually the random approach will always run until reaching 1800 time limit
        systest_ans[row_p][col_p * width + 3] = f'{num_random}'
        systest_ans[row_p][col_p * width + 4] = f'{time_random / len(seeds):.2f}'

        tot_systest_ranum += num_debarus
        tot_systest_random += num_random
        tot_systest_ranum_time += time_debarus / len(seeds)
        tot_systest_random_time += time_random / len(seeds)


        # 3: random_p_ranum
        num_random_p_ranum = 0
        time_random_p_ranum = 0.
        for seed in seeds:
            with open(f'results/training_inst_gen/grist/{ver_code}random_p_debarus/{seed}/stats/{mname}/data.json', 'r') as f:
                data = json.load(f)
            inference_stat = random_inference_status[seed][mname][err_node]
            time_random_p_ranum += inference_stat['tot_time']
            if data[err_node]['success']:
                num_random_p_ranum += 1
                time_random_p_ranum += data[err_node]['time']
        tot_systest_random_p_ranum += num_random_p_ranum
        tot_systest_random_p_ranum_time += time_random_p_ranum / len(seeds)

        # 4: ranum_p_random
        num_ranum_p_random = 0
        time_ranum_p_random = 0.
        for seed in seeds:
            with open(f'results/training_inst_gen/grist/{ver_code}debarus_p_random/{seed}/stats/{mname}/data.json', 'r') as f:
                data = json.load(f)
            inference_stat = ranum_inference_status[seed][mname][err_node]
            time_ranum_p_random += inference_stat['time']
            if data[err_node]['success']:
                num_ranum_p_random += 1
                time_ranum_p_random += data[err_node]['time']
            else:
                time_ranum_p_random += 1800. # time limit is 1800, actually the random approach will always run until reaching 1800 time limit
        tot_systest_ranum_p_random += num_ranum_p_random
        tot_systest_ranum_p_random_time += time_ranum_p_random / len(seeds)

        row_p += 1

        if row_p >= np.ceil((len(ordering) + 1) / col):
            row_p = 0
            col_p += 1

    systest_ans[row_p][col_p * width] = f'\\textbf{{Tot}}'
    systest_ans[row_p][col_p * width + 1] = '\\textbf{' + f'{tot_systest_ranum}' + '}'
    systest_ans[row_p][col_p * width + 2] = '\\textbf{' + f'{tot_systest_ranum_time / len(ordering):.2f}' + '}'
    systest_ans[row_p][col_p * width + 3] = '\\textbf{' + f'{tot_systest_random}' + '}'
    systest_ans[row_p][col_p * width + 4] = '\\textbf{' + f'{tot_systest_random_time / len(ordering):.2f}' + '}'

    for i, item in enumerate(systest_ans):
        if i == row_p:
            print(f'\\cline{{{col_p * width + 1}-{col_p * width + width}}}')
        print(' & '.join(item) + ' \\\\')


    print('')
    print('RANUM:', tot_systest_ranum, 'time=', tot_systest_ranum_time / len(ordering))
    print('Random:', tot_systest_random, 'time=', tot_systest_random_time / len(ordering))
    print('Random + RANUM:', tot_systest_random_p_ranum, 'time=', tot_systest_random_p_ranum_time / len(ordering))
    print('RANUM + random:', tot_systest_ranum_p_random, 'time=', tot_systest_ranum_p_random_time / len(ordering))


    # =============== Precondition-Fix Table =================
    print('=' * 20, 'Precondition-Fix', '=' * 20)

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


    for i, mname in enumerate(ordering):

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

    print('precond: debarus weight + input', tot_precond_debar_weight_input, 'time', tot_precond_debar_weight_input_time)
    print('precond: dexpand weight + input', tot_precond_expand_weight_input, 'time', tot_precond_expand_weight_input_time)
    print('precond: gd weight + input     ', tot_precond_gd_weight_input, 'time', tot_precond_gd_weight_input_time)
    print('precond: debarus weight        ', tot_precond_debar_weight, 'time', tot_precond_debar_weight_time)
    print('precond: dexpand weight        ', tot_precond_expand_weight, 'time', tot_precond_expand_weight_time)
    print('precond: gd weight             ', tot_precond_gd_weight, 'time', tot_precond_gd_weight_time)
    print('precond: debarus input         ', tot_precond_debar_input, 'time', tot_precond_debar_input_time)
    print('precond: dexpand input         ', tot_precond_expand_input, 'time', tot_precond_expand_input_time)
    print('precond: gd input              ', tot_precond_gd_input, 'time', tot_precond_gd_input_time)


    exit(0)