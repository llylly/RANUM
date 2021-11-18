"""
    This script runs whole benchmark set only for the purpose of certifying against numerical bugs
    It then records the detailed status and running time statistics
"""
import os
import time
import json
import pickle
from collections import OrderedDict

from interp.interp_module import load_onnx_from_file
from interp.interp_utils import AbstractionInitConfig, PossibleNumericalError

gt = {
    "1": [
        "Log:0"
    ],
    "10": [
        "Log:0"
    ],
    "11a": [
        "Exp_1:0",
        "truediv:0"
    ],
    "11b": [
        "Exp_1:0",
        "truediv:0"
    ],
    "11c": [
        "Exp_1:0",
        "truediv:0",
        "Log:0"
    ],
    "12": [
        "10"
    ],
    "13": [
        "2",
        "11"
    ],
    "14": [
        "Log:0"
    ],
    "15": [
        "Log:0"
    ],
    "16a": [
        "Exp_1:0",
        "truediv:0"
    ],
    "16b": [
        "Exp:0",
        "truediv:0",
        "Log:0"
    ],
    "16c": [
        "Exp:0",
        "truediv:0"
    ],
    "17": [
        "Exp_2:0",
        "truediv_14:0"
    ],
    "18": [
        "loss/Log:0"
    ],
    "19": [
        "Log:0"
    ],
    "20": [
        "Log:0"
    ],
    "21": [
        "Log:0"
    ],
    "22": [
        "18",
        "7"
    ],
    "23": [
        "10"
    ],
    "24": [
        "Log:0"
    ],
    "25": [
        "Log:0"
    ],
    "26": [
        "9",
        "15"
    ],
    "27": [
        "10"
    ],
    "28a": [
        "LogSoftmax__96:0"
    ],
    "28b": [
        "LogSoftmax__98:0"
    ],
    "28c": [
        "div_2:0",
        "div_4:0",
        "div_7:0",
        "div_10:0",
        "LogSoftmax__98:0"
    ],
    "28d": [
        "div_3:0",
        "div_6:0",
        "div_9:0",
        "LogSoftmax__98:0"
    ],
    "28e": [
        "mydiv4:0",
        "mydiv1_1:0",
        "mydiv4_1:0",
        "LogSoftmax__98:0"
    ],
    "29": [
        "dropout/truediv:0",
        "Log:0"
    ],
    "2a": [
        "cost/Log:0",
        "cost/Log_1:0"
    ],
    "2b": [
        "cost/Log:0",
        "cost/Log_1:0"
    ],
    "3": [
        "Log:0"
    ],
    "30": [
        "Log:0"
    ],
    "31": [
        "Log:0",
        "Log_1:0"
    ],
    "32": [
        "15"
    ],
    "33": [
        "7",
        "13"
    ],
    "34": [
        "16"
    ],
    "35a": [
        "Exp:0"
    ],
    "35b": [
        "Sqrt:0"
    ],
    "35c": [
        "Sqrt:0",
        "Log_1:0"
    ],
    "36a": [
        "Sqrt:0",
        "Log_1:0"
    ],
    "36b": [
        "Exp:0",
        "Sqrt:0",
        "Log_1:0"
    ],
    "36c": [
        "Sqrt:0"
    ],
    "37": [
        "6",
        "12"
    ],
    "38": [
        "6",
        "12"
    ],
    "39a": [
        "Sqrt:0",
        "Log_1:0"
    ],
    "39b": [
        "Exp_1:0",
        "Sqrt:0",
        "Log_1:0"
    ],
    "39c": [
        "Sqrt:0"
    ],
    "4": [
        "7"
    ],
    "40": [
        "Log:0"
    ],
    "41": [
        "Log:0"
    ],
    "42": [
        "loss/Log:0"
    ],
    "43a": [
        "Sqrt:0",
        "Log_1:0"
    ],
    "43b": [
        "Exp:0",
        "Sqrt:0",
        "Log_1:0"
    ],
    "43c": [
        "Sqrt:0"
    ],
    "44": [
        "Log:0"
    ],
    "45a": [
        "Log:0"
    ],
    "45b": [
        "Log:0"
    ],
    "46": [
        "10"
    ],
    "47": [
        "15"
    ],
    "48a": [
        "Log:0"
    ],
    "48b": [
        "Log:0"
    ],
    "49a": [
        "Sqrt:0",
        "Log_1:0"
    ],
    "49b": [
        "Exp:0",
        "Sqrt:0",
        "Log_1:0"
    ],
    "49c": [
        "Sqrt:0"
    ],
    "5": [
        "37"
    ],
    "50": [
        "Log:0"
    ],
    "51": [
        "44",
        "55"
    ],
    "52": [
        "Log:0"
    ],
    "53": [
        "7",
        "13"
    ],
    "54": [
        "6"
    ],
    "55": [
        "Log:0"
    ],
    "56": [
        "7",
        "13"
    ],
    "57": [
        "12"
    ],
    "58": [    "Log:0"
               ],
    "59": [
        "Log:0"
    ],
    "6": [
        "Log:0"
    ],
    "60": [
        "Log:0"
    ],
    "61": [
        "Log:0"
    ],
    "62": [
        "42",
        "56"
    ],
    "63": [
        "42",
        "56"
    ],
    "7": [
        "Log:0"
    ],
    "8": [
        "Log:0"
    ],
    "9a": [
        "Log:0",
        "Log_1:0"
    ],
    "9b": [
        "Log:0",
        "Log_1:0"
    ]
}

if __name__ == '__main__':
    PossibleNumericalError.continue_prop = True
    global_unsupported_ops = dict()
    print('on GRIST bench')
    nowdir = 'model_zoo/grist_protobufs_onnx'
    files = sorted([x for x in os.listdir(nowdir) if x.endswith('.onnx')])
    nowlen = len(files)

    for id, file in enumerate(files):
        print(f'[{id+1} / {nowlen}] {file}')
        stime = time.time()

        model = load_onnx_from_file(os.path.join(nowdir, file),
                                    customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
        loadtime = time.time()

        res = model.analyze(model.gen_abstraction_heuristics(file), {'average_pool_mode': 'coarse'})
        analyzetime = time.time()

        runningtime_stat = {'load': loadtime - stime, 'analyze': analyzetime - loadtime, 'all': analyzetime - stime}
        bugs = res
        unsupported_ops = list(model.unimplemented_types)

        print(f'* Time: load - {runningtime_stat["load"]:.3f} s | analyze - {runningtime_stat["analyze"]:.3f} s | all - {runningtime_stat["all"]:.3f} s')
        if len(bugs) > 0:
            print(f'* Found {len(bugs)} numerical bugs')
        if len(unsupported_ops) > 0:
            global_unsupported_ops[f'grist/{file}'] = unsupported_ops
            print(f'* Unsupported ops ({len(unsupported_ops)}): {unsupported_ops}')

        print(runningtime_stat)
        print(bugs.keys())

        for errop in bugs.keys():
            assert errop in gt[file[:-5]]
        assert len(bugs.keys()) == len(gt[file[:-5]])


        # pkl_package = {'time_stat': runningtime_stat, 'numerical_bugs': bugs, 'unspported_ops': unsupported_ops, 'model': model}
        # pkl_path = f'results/bug_verifier/grist/{file[:-5]}.pkl'
        # with open(pkl_path, 'wb') as f:
        #     pickle.dump(pkl_package, f)
        # print(f'saved to {pkl_path}')

    assert len(global_unsupported_ops) == 0

    print('grist error detection test passed')