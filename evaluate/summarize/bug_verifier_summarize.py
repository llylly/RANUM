"""
    Summarize the output of bug verifier
"""
import os
import pickle

# should be grist and/or debar
benchmarks = ['grist']

if __name__ == '__main__':
    for bench in benchmarks:
        print(f'Now on {bench}')
        nowdir = f'results/bug_verifier/{bench}'
        files = [item for item in os.listdir(nowdir) if item.endswith('.pkl')]
        files = sorted(files)
        for file in files:
            with open(nowdir + '/' + file, 'rb') as f:
                data = pickle.load(f)
            print(file)
            for k, v in data['numerical_bugs'].items():
                print(f'- On tensor {k} triggered by operator {v[1]}:')
                for item in v[0]:
                    print(str(item))
            runningtime_stat = data['time_stat']
            print(f'* Time: load - {runningtime_stat["load"]:.3f} s | analyze - {runningtime_stat["analyze"]:.3f} s | all - {runningtime_stat["all"]:.3f} s')
            print('')

