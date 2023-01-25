"""
    Baseline approach: gradient descent
    This approach reads in the final span of applying RANUM.
    Directly using this span and try to leverage gradient descent to generate failure-exhibiting intervals
"""


DEFAULT_LR = 1
DEFAULT_LR_DECAY = 0.1
DEFAULT_ITERS = 100
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
import numpy as np

import torch


from interp.interp_module import load_onnx_from_file
from interp.specified_vars import nospan_vars
from interp.specified_vars import nodiff_vars as nodiff_vars_default
from trigger.inference.robust_inducing_inst_gen import InducingInputGenModule
from trigger.hints import hard_coded_input_output_hints, customized_lr_inference_inst_gen

from evaluate.seeds import seeds

TLE = 300

# whitelist = ['1']
whitelist = []
blacklist = []

stime = time.time()
def prompt(msg):
    print(f'[{time.time() - stime:.3f}s] ' + msg)

def static_gradient_descent_inference_inst_gen(modelpath, mode, seed, span_dict, dumping_folder=None, default_lr=1., default_lr_decay=0.1, default_step=500,
                                               max_iters=100, span_len_step=np.sqrt(10), span_len_start=1e-10/2.0,
                                               skip_stages=list()):
    """
        The wrapped function for inference time input generation without span increasing and warm-up
    :param modelpath:
    :param mode: 'rand-input-rand-weight' / 'spec-input-rand-weight' / 'spec-input-spec-weight' / 'all'
    :return:
    """

    model = load_onnx_from_file(modelpath,
                                customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
    prompt('model initialized')

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

    stats_dict = dict()

    # store original input abstraction for clipping
    vanilla_lb_ub = dict()
    for s, abst in model.initial_abstracts.items():
        vanilla_lb_ub[s] = (torch.min(abst.lb).item(), torch.max(abst.ub).item())
    # print(vanilla_lb_ub)

    # scan all error points
    err_nodes, err_exceps = list(), list()
    for k, v in initial_errors.items():
        error_entities, root, catastro = v
        if not catastro:
            for error_entity in error_entities:
                err_nodes.append(error_entity.var_name)
                err_exceps.append(error_entity)

    if not os.path.exists(os.path.join(dumping_folder, bare_name)):
        os.makedirs(os.path.join(dumping_folder, bare_name))

    err_seq_no = 0
    for err_node, err_excep in zip(err_nodes, err_exceps):
        stime = time.time()

        now_stats_item = dict()
        success = False
        category = None

        if (mode == 'rand-input-rand-weight' or mode == 'all') and 'rand-input-rand-weight' not in skip_stages:
            print('rand-input-rand-weight stage')
            ls_time = time.time()
            inputgen_module = InducingInputGenModule(model, seed=seed, no_diff_vars=nodiff_vars_default, no_span_vars=nospan_vars,
                                                     from_init_dict=True,
                                                     span_len=0.)
            loss, errors = inputgen_module.forward([err_node], [err_excep])
            trigger_nodes, robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)
            if err_node in trigger_nodes:
                # means random input and random weights can trigger the error
                category = 'rand-input-rand-weight'

                # try to maximize span length
                span_len = span_dict[err_node].get('span_len', 1e-5)
                print('loaded span len =', span_len)
                inputgen_module.set_span_len(span_len, no_span_vars=nospan_vars)
                inputgen_module.clip_to_valid_range(vanilla_lb_ub)
                loss, errors = inputgen_module.forward([err_node], [err_excep])
                trigger_nodes, robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)
                if err_node not in trigger_nodes:
                    pass
                else:
                    success = True
                now_stats_item['span_len'] = span_len

                if dumping_folder is not None:
                    if not os.path.exists(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))):
                        os.makedirs(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt')))
                    torch.save(inputgen_module.dump_init_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))
                    torch.save(inputgen_module.dump_init_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_gen.pt'))

            lt_time = time.time()
            now_stats_item['rand-input-rand-weight-time'] = lt_time - ls_time

        if (mode == 'spec-input-rand-weight' or (mode == 'all' and not success)) and 'spec-input-rand-weight' not in skip_stages:
            print('spec-input-rand-weight stage')
            # detect input nodes, then other nodes are weights and they will be fixed
            ls_time = time.time()
            iters_log = dict()

            # extract model name from path
            if bare_name in hard_coded_input_output_hints:
                input_nodes, loss_function_nodes = hard_coded_input_output_hints[bare_name][0].copy(), hard_coded_input_output_hints[bare_name][1].copy()
            else:
                input_nodes, loss_function_nodes = model.detect_input_and_output_nodes(alert_exceps=False)
            no_diff_vars = nodiff_vars_default + [s for s in model.initial_abstracts if s not in input_nodes]


            inputgen_module = InducingInputGenModule(model, seed=seed, no_diff_vars=no_diff_vars, no_span_vars=nospan_vars,
                                                     from_init_dict=True,
                                                     span_len=0.)

            # try to maximize span length
            span_len = -1.
            try_span_len = span_dict[err_node].get('span_len', 1e-5)

            print('loaded span len =', try_span_len)
            now_span_len_success = False

            inputgen_module.set_span_len(try_span_len, no_span_vars=nospan_vars)
            optimizer = torch.optim.Adam(inputgen_module.parameters(), lr=default_lr if bare_name not in customized_lr_inference_inst_gen else customized_lr_inference_inst_gen[bare_name])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=default_step)

            for iter in range(max_iters):
                # print('----------------')

                optimizer.zero_grad()
                loss, errors = inputgen_module.forward([err_node], [err_excep])
                trigger_nodes, robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)

                print(f'[{bare_name},err{err_seq_no},{err_node}] iter =', iter, 'loss =', loss, 'has robust error =', len(robust_errors))

                if err_node in trigger_nodes:
                    now_span_len_success = True
                    print('robust error found!')
                    break
                try:
                    loss.backward()
                except:
                    break
                optimizer.step()
                scheduler.step()

                inputgen_module.clip_to_valid_range(vanilla_lb_ub)

                if time.time() - ls_time >= TLE: break

            iters_log[try_span_len] = iter

            if now_span_len_success:
                success = True
                category = 'spec-input-rand-weight'
                span_len = try_span_len
                now_stats_item['span_len'] = span_len

                if dumping_folder is not None:
                    if not os.path.exists(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))):
                        os.makedirs(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt')))
                    torch.save(inputgen_module.dump_init_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))
                    torch.save(inputgen_module.dump_gen_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_gen.pt'))


            lt_time = time.time()
            now_stats_item['spec-input-rand-weight-time'] = lt_time - ls_time
            now_stats_item['spec-input-rand-weight-iters'] = iters_log

        if (mode == 'spec-input-spec-weight' or (mode == 'all' and not success)) and 'spec-input-spec-weight' not in skip_stages:
            print('spec-input-spec-weight stage')

            ls_time = time.time()
            iters_log = dict()

            inputgen_module = InducingInputGenModule(model, seed=seed, no_diff_vars=nodiff_vars_default, no_span_vars=nospan_vars,
                                                     from_init_dict=True,
                                                     span_len=0.)

            # try to maximize span length
            span_len = -1.
            try_span_len = span_dict[err_node].get('span_len', 1e-5)

            print('loaded span len =', try_span_len)
            now_span_len_success = False

            inputgen_module.set_span_len(try_span_len, no_span_vars=nospan_vars)
            optimizer = torch.optim.Adam(inputgen_module.parameters(), lr=default_lr if bare_name not in customized_lr_inference_inst_gen else customized_lr_inference_inst_gen[bare_name])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=default_step)

            for iter in range(max_iters):
                # print('----------------')

                optimizer.zero_grad()
                loss, errors = inputgen_module.forward([err_node], [err_excep])
                trigger_nodes, robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)

                print(f'[{bare_name},err{err_seq_no},{err_node}] iter =', iter, 'loss =', loss, 'has robust error =', len(robust_errors))

                if err_node in trigger_nodes:
                    now_span_len_success = True
                    print('robust error found!')
                    break

                try:
                    loss.backward()
                except:
                    break
                optimizer.step()
                scheduler.step()

                inputgen_module.clip_to_valid_range(vanilla_lb_ub)

                if time.time() - ls_time >= TLE: break

            iters_log[try_span_len] = iter

            if now_span_len_success:
                success = True
                category = 'spec-input-spec-weight'
                span_len = try_span_len
                now_stats_item['span_len'] = span_len


                if dumping_folder is not None:
                    if not os.path.exists(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))):
                        os.makedirs(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt')))
                    torch.save(inputgen_module.dump_init_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))
                    torch.save(inputgen_module.dump_gen_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_gen.pt'))

            lt_time = time.time()
            now_stats_item['spec-input-spec-weight-time'] = lt_time - ls_time
            now_stats_item['spec-input-spec-weight-iters'] = iters_log

        now_stats_item['success'] = success
        now_stats_item['category'] = category
        now_stats_item['time'] = time.time() - stime
        now_stats_item['err_seq_no'] = err_seq_no
        stats_dict[err_node] = now_stats_item

        err_seq_no += 1

    return stats_dict





if __name__ == '__main__':

    for seed in seeds:
        print('*' * 20)
        print('*' * 20)
        print('seed =', seed)

        shorten_statuses = dict()

        for benchmark in run_benchmarks:

            if benchmark == 'grist':
                print('on GRIST bench')
                nowdir = 'model_zoo/grist_protobufs_onnx'
            elif benchmark == 'debar':
                print('on DEBAR bench')
                nowdir = 'model_zoo/tf_protobufs_onnx'
            files = sorted([x for x in os.listdir(nowdir) if x.endswith('.onnx')])
            nowlen = len(files)


            data_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/baseline/gradient_descent/data'
            orig_stats_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/stats'
            stats_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/baseline/gradient_descent/stats'
            print('data_dir =', data_dir, 'stats_dir =', stats_dir)

            statuses = dict()

            nopass = False

            for id, file in enumerate(files):
                print(f'[{id+1} / {nowlen}] {file}')
                barefilename = file.rsplit('.', maxsplit=1)[0]

                if len(whitelist) > 0 and barefilename not in whitelist: continue
                if barefilename in blacklist: continue

                # read spans from status dict
                with open(os.path.join(orig_stats_dir, barefilename, 'data.json'), 'r') as f:
                    stats = json.load(f)
                # print(stats)

                status = static_gradient_descent_inference_inst_gen(os.path.join(nowdir, file), 'all', seed, stats, data_dir,
                                                     default_lr=DEFAULT_LR if barefilename not in customized_lr_inference_inst_gen
                                                     else customized_lr_inference_inst_gen[barefilename],
                                                     default_lr_decay=DEFAULT_LR_DECAY, default_step=DEFAULT_STEP, max_iters=DEFAULT_ITERS,
                                                     skip_stages=skip_stages_by_name[barefilename] if barefilename in skip_stages_by_name else list())


                print(json.dumps(status, indent=2))
                statuses[barefilename] = status
                if not os.path.exists(os.path.join(stats_dir, barefilename)):
                    os.makedirs(os.path.join(stats_dir, barefilename))
                with open(os.path.join(stats_dir, barefilename, f'data.json'), 'w') as f:
                    json.dump(status, f, indent=2)

                statuses[barefilename] = status
                for k, v in status.items():
                    shorten_statuses[barefilename+':'+k] = {
                        'success': v['success'],
                        'category': v['category']
                    }

        print('Tot Cases:', len(statuses))
        print(json.dumps(shorten_statuses, indent=2))




        #
        #         status = inducing_inference_inst_gen(os.path.join(nowdir, file), 'all', seed, data_dir,
        #                                              default_lr=DEFAULT_LR if barefilename not in customized_lr_inference_inst_gen
        #                                              else customized_lr_inference_inst_gen[barefilename],
        #                                              default_lr_decay=DEFAULT_LR_DECAY, default_step=DEFAULT_STEP, max_iters=DEFAULT_ITERS,
        #                                              skip_stages=skip_stages_by_name[barefilename] if barefilename in skip_stages_by_name else list())
        #
        #
        #         print(json.dumps(status, indent=2))
        #         statuses[barefilename] = status
        #         if not os.path.exists(os.path.join(stats_dir, barefilename)):
        #             os.makedirs(os.path.join(stats_dir, barefilename))
        #         with open(os.path.join(stats_dir, barefilename, f'data.json'), 'w') as f:
        #             json.dump(status, f, indent=2)
        #
        #         statuses[barefilename] = status
        #         for k, v in status.items():
        #             shorten_statuses[barefilename+':'+k] = {
        #                 'success': v['success'],
        #                 'category': v['category']
        #             }
        #
        # print('Tot Cases:', len(statuses))
        # print(json.dumps(shorten_statuses, indent=2))