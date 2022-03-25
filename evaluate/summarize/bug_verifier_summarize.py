"""
    Summarize the output of bug verifier
"""
import os
import pickle
import json

# should be grist and/or debar
benchmarks = ['grist']

# account for different tries
summary_no = 2

if __name__ == '__main__':
    for bench in benchmarks:
        print(f'Now on {bench}')
        nowdir = f'results/bug_verifier/{bench}'
        output_js_path = f'results/summary/bug_verifier_{bench}_{summary_no}.json'

        out = dict()

        files = [item for item in os.listdir(nowdir) if item.endswith('.pkl')]
        files = sorted(files)
        for file in files:
            with open(nowdir + '/' + file, 'rb') as f:
                data = pickle.load(f)
            print(file)
            bugcnt = 0
            for k, v in data['numerical_bugs'].items():
                print(f'- On tensor {k} triggered by operator {v[1]}:')
                for item in v[0]:
                    bugcnt += 1
                    print(str(item))
            runningtime_stat = data['time_stat']
            print(f'* Time: load - {runningtime_stat["load"]:.3f} s | analyze - {runningtime_stat["analyze"]:.3f} s | all - {runningtime_stat["all"]:.3f} s')
            print('')

            out[file[:-4]] = dict()
            out[file[:-4]]['bugnum'] = bugcnt
            out[file[:-4]]['timeload'] = runningtime_stat["load"]
            out[file[:-4]]['timeanalyze'] = runningtime_stat["analyze"]
            out[file[:-4]]['timeall'] = runningtime_stat["all"]

        print('summarize to', output_js_path)
        with open(output_js_path, 'w') as f:
            json.dump(out, f, indent=2)

