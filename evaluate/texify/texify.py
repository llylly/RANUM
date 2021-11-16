import os
import json

# mode = bug_verifier / 'precond_gen
mode = 'bug_verifier'
# mode = 'precond_gen'
# dataset = grist / debar
dataset = 'grist'
# replica = 1/2/3/4/5
replica = 1


if __name__ == '__main__':
    file = f'results/summary/{mode}_{dataset}_{replica}.json'
    with open(file, 'r') as f:
        data = json.load(f)
    if dataset == 'grist':
        keys = sorted(data.keys(), key=lambda x: int(str(x).rstrip('abcdefghijklmnopqrstuvwxyz')) + ord(x[-1]) / 255.)
    else:
        keys = list(data.keys())
    if mode == 'bug_verifier':
        fields = ['bugnum', 'timeanalyze']
        names = ['\# Bugs', 'Analyze Time (s)']
    if mode == 'precond_gen':
        fields = ['time_stat/precond', 'success', 'precond_stat/tot_start_points', 'precond_stat/tot_changed_nodes', 'precond_stat/average_shrinkage']
        names = ['Pre-Cond. Gen. Time (s)', 'Success', '\# Input/Param Nodes', '\# Constrained Nodes', 'Average Interval Length Shrinkage']
    text = '\\begin{tabular}{' + 'c' * (len(fields) + 1) + '}\n'
    text += '\\toprule \n'
    text += '  ' + ' & '.join(['No.'] + names) + ' \\\\\n'
    text += '\\midrule \n'

    for k in keys:
        row = '  ' + k
        for now_field in fields:
            row += ' & '
            now_fields = now_field.split('/')
            now_item = data[k]
            for field in now_fields:
                now_item = now_item[field]
            print(type(now_item))
            if isinstance(now_item, float):
                row += f'${now_item:.3f}$'
            elif (isinstance(now_item, str) and now_item == 'True') or (isinstance(now_item, bool) and now_item):
                row += '$\\checkmark$'
            elif isinstance(now_item, int):
                row += f'${now_item}$'
        row += ' \\\\\n'
        text += row

    text += '\\bottomrule \n'
    text += '\\end{tabular}'
    print(text)
    with open('results/texify/nowtable.tex', 'w') as f:
        f.write(text)
