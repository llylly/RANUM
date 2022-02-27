"""
    This script runs whole benchmark set only for the purpose of generating error-inducing inference-time inputs and model weights.
    It then records the detailed status and running time statistics.
    It also records the initial inputs & weights and error-inducing inputs & weights
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

ver_code = f'v3_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'
# v4 changes the loss formulation
# ver_code = f'v4_lr{DEFAULT_LR}_decay_{DEFAULT_LR_DECAY}_step{DEFAULT_STEP}_iter{DEFAULT_ITERS}'

import os
import time
import json

import torch

from interp.interp_module import load_onnx_from_file
from interp.specified_vars import nodiff_vars, nospan_vars
from trigger.inference.robust_inducing_inst_gen import InducingInputGenModule, inducing_inference_inst_gen
from trigger.hints import customized_lr_inference_inst_gen
from evaluate.seeds import seeds

# whitelist = ['1']
whitelist = []
blacklist = []
# blacklist = ['17']

if __name__ == '__main__':
    for seed in seeds[:1]:
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


            data_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/data'
            stats_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/stats'
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)
            print('data_dir =', data_dir, 'stats_dir =', stats_dir)

            statuses = dict()

            nopass = False

            for id, file in enumerate(files):
                print(f'[{id+1} / {nowlen}] {file}')
                barefilename = file.rsplit('.', maxsplit=1)[0]
                if len(whitelist) > 0 and barefilename not in whitelist: continue
                if barefilename in blacklist: continue
                if not nopass:
                    passornot = input('pass?')
                    if passornot == 'Y':
                        continue
                    else:
                        nopass = True
                status = inducing_inference_inst_gen(os.path.join(nowdir, file), 'all', seed, data_dir,
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


        # global_unsupported_ops = dict()
        # statuses = dict()
        #
        # for benchmark in run_benchmarks:
        #
        #     if benchmark == 'grist':
        #         print('on GRIST bench')
        #         nowdir = 'model_zoo/grist_protobufs_onnx'
        #     elif benchmark == 'debar':
        #         print('on DEBAR bench')
        #         nowdir = 'model_zoo/tf_protobufs_onnx'
        #     files = sorted([x for x in os.listdir(nowdir) if x.endswith('.onnx')])
        #     nowlen = len(files)
        #
        #     data_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/data'
        #     stats_dir = f'results/inference_inst_gen/{benchmark}/{ver_code}/{seed}/stats'
        #     if not os.path.exists(data_dir):
        #         os.makedirs(data_dir)
        #     if not os.path.exists(stats_dir):
        #         os.makedirs(stats_dir)
        #     print('data_dir =', data_dir, 'stats_dir =', stats_dir)
        #
        #
        #     for id, file in enumerate(files):
        #         print(f'[{id+1} / {nowlen}] {file}')
        #         barefilename = file.split('.')[0]
        #         if len(whitelist) > 0 and barefilename not in whitelist: continue
        #
        #         stime = time.time()
        #
        #         model = load_onnx_from_file(os.path.join(nowdir, file),
        #                                     customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
        #         loadtime = time.time()
        #
        #         initial_errors = model.analyze(model.gen_abstraction_heuristics(barefilename), {'average_pool_mode': 'coarse', 'diff_order': 1})
        #         if len(initial_errors) == 0:
        #             print('No numerical bug')
        #         else:
        #             print(f'{len(initial_errors)} possible numerical bug(s)')
        #             for k, v in initial_errors.items():
        #                 print(f'- On tensor {k} triggered by operator {v[1]}:')
        #                 for item in v[0]:
        #                     print(str(item))
        #         analyzetime = time.time()
        #
        #         # catch unsupported ops
        #         unsupported_ops = list(model.unimplemented_types)
        #         if len(unsupported_ops) > 0:
        #             global_unsupported_ops[f'{benchmark}/{file}'] = unsupported_ops
        #             print(f'* Unsupported ops ({len(unsupported_ops)}): {unsupported_ops}')
        #
        #         # exercise all error points
        #         err_nodes, err_exceps = list(), list()
        #         for k, v in initial_errors.items():
        #             error_entities, root, catastro = v
        #             if not catastro:
        #                 for error_entity in error_entities:
        #                     err_nodes.append(error_entity.var_name)
        #                     err_exceps.append(error_entity)
        #
        #         inputgen_module = InducingInputGenModule(model, seed=seed, no_diff_vars=nodiff_vars, no_span_vars=nospan_vars,
        #                                                  from_init_dict=True,
        #                                                  span_len=DEFAULT_SPAN if barefilename not in customize_span else customize_span[barefilename])
        #         for errno, (err_node, err_excep) in enumerate(zip(err_nodes, err_exceps)):
        #             gen_stime = time.time()
        #             success = False
        #             inputgen_module.rewind()
        #
        #             optimizer = torch.optim.Adam(inputgen_module.parameters(), lr=DEFAULT_LR if barefilename not in customized_lr_inference_inst_gen else customized_lr_inference_inst_gen[barefilename])
        #             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DEFAULT_STEP)
        #
        #             for iter in range(DEFAULT_ITERS):
        #                 # print('----------------')
        #
        #                 optimizer.zero_grad()
        #                 loss, errors = inputgen_module.forward([err_node], [err_excep])
        #                 robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)
        #
        #                 print(f'[{id+1},err{errno}] iter =', iter, 'loss =', loss, 'has robust error =', len(robust_errors))
        #
        #                 if len(robust_errors) > 0:
        #                     success = True
        #                     print('robust error found!')
        #                     break
        #
        #                 loss.backward()
        #                 optimizer.step()
        #                 scheduler.step()
        #             gen_ttime = time.time()
        #
        #             status = {'analyzetime': analyzetime - loadtime, 'loadtime': loadtime - stime, 'gentime': gen_ttime - gen_stime,
        #                       'success': success, 'errnode': err_node, 'erroptype': err_excep.optype, 'iters': iter}
        #             if success:
        #                 details = inputgen_module.robust_input_study(False)
        #                 status['details'] = details
        #
        #             # print status
        #             print(json.dumps(status, indent=2))
        #             statuses[barefilename] = status
        #             if not os.path.exists(os.path.join(stats_dir, barefilename)):
        #                 os.makedirs(os.path.join(stats_dir, barefilename))
        #             with open(os.path.join(stats_dir, barefilename, f'{errno}.json'), 'w') as f:
        #                 json.dump(status, f, indent=2)
        #
        #             if success:
        #                 if not os.path.exists(os.path.join(data_dir, barefilename)):
        #                     os.makedirs(os.path.join(data_dir, barefilename))
        #                 torch.save(inputgen_module.dump_init_weights(), os.path.join(data_dir, barefilename, f'{errno}_init.pt'))
        #                 torch.save(inputgen_module.dump_gen_weights(), os.path.join(data_dir, barefilename, f'{errno}_gen.pt'))
        #             else:
        #                 # by default will fail on 28c, 28d, 28e
        #                 if barefilename not in ['28c', '28d', '28e']:
        #                     raise Exception(f'Not succeessful at [{id+1},err{errno}]: {file}')
        #
        #
        # print('Unsupported Ops:', global_unsupported_ops)
        # print('Tot Cases:', len(statuses))
        # print(json.dumps(statuses, indent=2))
