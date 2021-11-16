"""
    Summarize the output of bug verifier
"""
import os
import pickle
import json

# should be grist and/or debar
benchmarks = ['grist']

# account for different tries
summary_no = 1

if __name__ == '__main__':
    for bench in benchmarks:
        print(f'Now on {bench}')
        nowdir = f'results/precond_gen/{bench}'
        output_js_path = f'results/summary/precond_gen_{bench}_{summary_no}.json'

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

            out[file[:-4]] = data
            del out[file[:-4]]['numerical_bugs']
            out[file[:-4]]['bugnum'] = bugcnt

        print('summarize to', output_js_path)
        with open(output_js_path, 'w') as f:
            json.dump(out, f)

