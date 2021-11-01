"""
    This script runs whole benchmark set only for the purpose of certifying against numerical bugs
    It then records the detailed status and running time statistics
"""
import os
import time
import pickle

from interp.interp_module import load_onnx_from_file
from interp.interp_utils import AbstractionInitConfig

run_benchmarks = ['debar']

if __name__ == '__main__':
    global_unsupported_ops = dict()
    for bench_type in run_benchmarks:
        if bench_type == 'grist':
            print('on GRIST bench')
            nowdir = 'model_zoo/grist_protobufs_onnx'
        elif bench_type == 'debar':
            print('on DEBAR bench')
            nowdir = 'model_zoo/tf_protobufs_onnx'
        files = sorted([x for x in os.listdir(nowdir) if x.endswith('.onnx')])
        nowlen = len(files)
        for id, file in enumerate(files):
            print(f'[{id+1} / {nowlen}] {file}')
            stime = time.time()

            model = load_onnx_from_file(os.path.join(nowdir, file),
                                        customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
            loadtime = time.time()

            res = model.analyze(model.gen_abstraction_heuristics(), {'average_pool_mode': 'coarse'})
            analyzetime = time.time()

            runningtime_stat = {'load': loadtime - stime, 'analyze': analyzetime - loadtime, 'all': analyzetime - stime}
            bugs = res
            unsupported_ops = list(model.unimplemented_types)

            print(f'* Time: load - {runningtime_stat["load"]:.3f} s | analyze - {runningtime_stat["analyze"]:.3f} s | all - {runningtime_stat["all"]:.3f} s')
            if len(bugs) > 0:
                print(f'* Found {len(bugs)} numerical bugs')
            if len(unsupported_ops) > 0:
                global_unsupported_ops[f'{bench_type}/{file}'] = unsupported_ops
                print(f'* Unsupported ops ({len(unsupported_ops)}): {unsupported_ops}')

            pkl_package = {'time_stat': runningtime_stat, 'numerical_bugs': bugs, 'unspported_ops': unsupported_ops, 'model': model}
            pkl_path = f'results/bug_verifier/{bench_type}/{file[:-5]}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(pkl_package, f)
            print(f'saved to {pkl_path}')

    print('Unsupported Ops:', global_unsupported_ops)