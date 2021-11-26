"""
    This script tries to generate preconditon that secures the model from numerical bugs via gradient descent.
"""

EPS = 1e-5

import time
import argparse
import os

import torch
import torch.nn as nn
import torch.optim

import numpy as np

from interp.interp_operator import Abstraction
from interp.interp_module import load_onnx_from_file, InterpModule
from interp.interp_utils import PossibleNumericalError, parse_attribute
from trigger.inference.range_clipping import range_clipping

class PrecondGenModule(nn.Module):
    """
        The class for generating secure preconditon with gradient descent.
    """

    def __init__(self, model, no_diff_vars=list()):
        super(PrecondGenModule, self).__init__()
        # assure that the model is already analyzed
        assert model.abstracts is not None

        new_initial_abstracts = dict()

        # centers and scales are parameters
        self.centers = dict()
        self.scales = dict()
        self.spans = dict()

        self.model = model
        self.abstracts = dict()

        for s, orig in model.initial_abstracts.items():
            new_abst = self.construct(orig, s, s in no_diff_vars)
            self.abstracts[s] = new_abst

        no = 0
        for k, v in self.centers.items():
            self.register_parameter(f'centers:{no}', v)
            no += 1
        for k, v in self.scales.items():
            self.register_parameter(f'scales:{no}', v)
            no += 1

    def forward(self, node, excep, div_branch='+'):
        """
            construct precondition generation loss
        :param node: the node name (str) or (list of that) that causes numerical error
        :param excep: the numerical error object or (list of that)
        :param div_branch: for div op, whether to use positive branch or negative branch, '+' or '-'
        :return:
        """
        self.abstracts.clear()
        for s, orig in self.model.initial_abstracts.items():
            new_abst = self.construct(orig, s)
            self.abstracts[s] = new_abst
        _, errors = self.model.forward(self.abstracts, {'diff_order': 1})

        loss = torch.tensor(0.)

        if not isinstance(node, list):
            nodes, exceps = [node], [excep]
        else:
            nodes, exceps = node, excep

        for node, excep in zip(nodes, exceps):
            prev_nodes = self.find_prev_nodes_optypes(node)
            if excep.optype == 'Log' or excep.optype == 'Sqrt':
                # print(node)
                prev_node = prev_nodes[0][0]
                prev_abs = self.abstracts[prev_node]
                loss += torch.sum(torch.where(prev_abs.lb <= PossibleNumericalError.UNDERFLOW_LIMIT, -prev_abs.lb, torch.zeros_like(prev_abs.lb)))
            elif excep.optype == 'Exp':
                prev_node = prev_nodes[0][0]
                prev_abs = self.abstracts[prev_node]
                thres = PossibleNumericalError.OVERFLOW_D * np.log(10)
                loss += torch.sum(torch.where(prev_abs.ub >= thres, prev_abs.ub - thres, torch.zeros_like(prev_abs.ub)))
            elif excep.optype == 'LogSoftMax':
                prev_node = prev_nodes[0][0]
                axis = parse_attribute(prev_nodes[0][2]).get('axis', -1)
                prev_abs = self.abstracts[prev_node]
                loss += torch.sum(torch.where(prev_abs.ub - torch.amin(prev_abs.lb, dim=axis, keepdim=True) >= PossibleNumericalError.OVERFLOW_D - PossibleNumericalError.UNDERFLOW_D,
                                              prev_abs.ub - torch.amin(prev_abs.lb, dim=axis, keepdim=True) - (PossibleNumericalError.OVERFLOW_D - PossibleNumericalError.UNDERFLOW_D),
                                              torch.zeros_like(prev_abs.lb)))
            elif excep.optype == 'Div':
                # contains zero in div
                divisor = [item for item in prev_nodes if item[3] == 1][0]
                divisor_abs = self.abstracts[divisor[0]]
                if div_branch == '+':
                    loss += torch.sum(torch.where((divisor_abs.lb <= PossibleNumericalError.UNDERFLOW_LIMIT) & (divisor_abs.ub >= -PossibleNumericalError.UNDERFLOW_LIMIT),
                                      -divisor_abs.lb, torch.zeros_like(divisor_abs.lb)))
                elif div_branch == '-':
                    loss += torch.sum(torch.where((divisor_abs.lb <= PossibleNumericalError.UNDERFLOW_LIMIT) & (divisor_abs.ub >= -PossibleNumericalError.UNDERFLOW_LIMIT),
                                      divisor_abs.ub, torch.zeros_like(divisor_abs.ub)))
            else:
                raise Exception(f'excep type {excep.optype} not supported yet, you can follow above template to extend the support')
        return loss, errors

    def grad_step(self, center_lr=0.1, scale_lr=0.1, min_step=0.1, regularizer=0.):
        """
            My customzed FGSM-style gradient step to avoid gradient explosion
        :param center_lr: relative learning rate for center parameters
        :param scale_lr: relative learning rate for scale parameters
        :param min_step: minimum step size for center, in case the center magnitude is too small
        :param regularizer: if scale < regularizer, stop to update to avoid too narrow range
        :return:
        """
        for k, v in self.centers.items():
            # print(k)
            # print(v.data)
            if v.grad is not None:
                v.data = v.data - center_lr * torch.maximum(torch.abs(v.data), torch.full_like(v.data, min_step)) * torch.sign(v.grad)

        for k, v in self.scales.items():
            # print(k)
            # print(v.data)
            # print((v.data > regularizer))
            # print(torch.any(v.data > regularizer))
            if v.grad is not None and torch.any(v.data > regularizer):
                v.data = v.data - scale_lr * torch.abs(v.data) * torch.sign(v.grad)

    def precondition_study(self, print_stdout=True):
        """
            summarize the heuristics of generated preconditions
        :param print: whether to print to console
        :return:
        """
        tot_nodes = len(self.model.start_points)
        changed_nodes = 0
        average_shrink = list()
        maximum_shrink = 0.
        minimum_shrink = 1.
        for s in self.model.start_points:
            if s == InterpModule.SUPER_START_NODE: continue
            orig_abst = self.model.abstracts[s]
            now_abst = self.abstracts[s]
            if self.tensor_equal(orig_abst.lb, now_abst.lb) and self.tensor_equal(orig_abst.ub, now_abst.ub):
                minimum_shrink = 0.
                average_shrink.append(0.)
            else:
                changed_nodes += 1
                now_shrinkage = self.compute_shrinkage(now_abst, orig_abst)
                maximum_shrink = max(maximum_shrink, now_shrinkage)
                minimum_shrink = min(minimum_shrink, now_shrinkage)
                average_shrink.append(now_shrinkage)
        ans = {
            'tot_start_points': tot_nodes,
            'tot_changed_nodes': changed_nodes,
            'average_shrinkage': np.mean(average_shrink),
            'maximum_shrinkage': maximum_shrink,
            'minimum_shrink': minimum_shrink
        }
        if print_stdout:
            print('tot start point', tot_nodes)
            print('tot changed nodes', changed_nodes)
            print('average shrinkage', np.mean(average_shrink))
            print('maximum shrinkage', maximum_shrink)
            print('minimum shrinkage', minimum_shrink)
        return ans



    def construct(self, original: Abstraction, var_name: str, no_diff=False):
        """
            From the original abstraction to create a new abstraction parameterized by span * scale and center:
            [center - span * scale, center + span * scale]
            where center and scale are learnable parameters
        :param original: original abstraction from the first run
        :return: parameterized abstraction
        """
        def _work(lb, ub, name):
            center = nn.Parameter((lb + ub) / 2., requires_grad=lb.requires_grad and (not no_diff))
            self.centers[name] = center
            if torch.sum(ub - lb) > EPS:
                # non-constant, add the span factor
                span = torch.tensor(center - lb, requires_grad=False)
                scale = nn.Parameter(torch.ones_like(span, dtype=span.dtype), requires_grad=not no_diff)
                self.spans[name] = span
                self.scales[name] = scale
                new_lb, new_ub = center - scale * span, center + scale * span
            else:
                new_lb, new_ub = center, center
            return new_lb, new_ub

        ans = Abstraction()
        if isinstance(original.lb, list):
            ans.lb = list()
            ans.ub = list()
            for i in range(len(original.lb)):
                if var_name + f'_{i}' in self.centers and var_name + f'_{i}' in self.spans:
                    item_lb = self.centers[var_name + f'_{i}'] - self.spans[var_name + f'_{i}'] * self.scales[var_name + f'_{i}']
                    item_ub = self.centers[var_name + f'_{i}'] + self.spans[var_name + f'_{i}'] * self.scales[var_name + f'_{i}']
                elif var_name + f'_{i}' in self.centers and var_name + f'_{i}' not in self.spans:
                    item_lb = self.centers[var_name + f'_{i}']
                    item_ub = self.centers[var_name + f'_{i}']
                else:
                    item_lb, item_ub = _work(original.lb[i], original.ub[i], var_name + f'_{i}')
                ans.lb.append(item_lb)
                ans.ub.append(item_ub)
        else:
            if var_name in self.centers and var_name in self.spans:
                ans.lb = self.centers[var_name] - self.spans[var_name] * self.scales[var_name]
                ans.ub = self.centers[var_name] + self.spans[var_name] * self.scales[var_name]
            elif var_name in self.centers and var_name not in self.spans:
                ans.lb = self.centers[var_name]
                ans.ub = self.centers[var_name]
            else:
                ans.lb, ans.ub = _work(original.lb, original.ub, var_name)
        ans.shape = original.shape.copy()
        ans.splits = original.splits.copy()
        ans.var_name = original.var_name
        return ans

    def find_prev_nodes_optypes(self, node):
        # (prev node name, prev node optype, prev node obj, prev node index in op)
        return sorted([(k, [item[3] for item in v if item[0] == node][0], [item[5] for item in v if item[0] == node][0],
                 min([item[1] for item in v if item[0] == node]))
                for k, v in self.model.edges.items() if any([item[0] == node for item in v])],
                      key=lambda x:x[3])

    @staticmethod
    def tensor_equal(a, b, EPS=1e-5):
        # tensor float equal
        if isinstance(a, list):
            return all([PrecondGenModule.tensor_equal(item_a, item_b) for item_a, item_b in zip(a, b)])
        else:
            return torch.norm(a.view(-1) - b.view(-1), p=float('inf')).detach().cpu().item() <= EPS

    @staticmethod
    def compute_shrinkage(a, b):
        # compute the shrinkage of abstraction a w.r.t. abstraction b

        def work(alb, aub, blb, bub):
            if PrecondGenModule.tensor_equal(blb, bub):
                return 0.0
            else:
                delta_b = bub - blb
                delta_a = aub - alb
                return 1.0 - torch.mean(delta_a / torch.clip(delta_b, min=1e-5)).detach().cpu().item()

        if isinstance(a.lb, list):
            return np.mean([work(a.lb[i], a.ub[i], b.lb[i], b.ub[i]) for i in range(len(a.lb))])
        else:
            return work(a.lb, a.ub, b.lb, b.ub)

# ================ # ================
# below is for individual instance running

parser = argparse.ArgumentParser()
parser.add_argument('modelpath', type=str, help='model architecture file path')

stime = time.time()
def prompt(msg):
    print(f'[{time.time() - stime:.3f}s] ' + msg)

if __name__ == '__main__':
    args = parser.parse_args()

    model = load_onnx_from_file(args.modelpath,
                                customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
    prompt('model initialized')

    initial_errors = model.analyze(model.gen_abstraction_heuristics(os.path.split(args.modelpath)[-1].split('.')[0]), {'average_pool_mode': 'coarse', 'diff_order': 1})
    prompt('analysis done')
    if len(initial_errors) == 0:
        print('No numerical bug')
    else:
        print(f'{len(initial_errors)} possible numerical bug(s)')
        for k, v in initial_errors.items():
            print(f'- On tensor {k} triggered by operator {v[1]}:')
            for item in v[0]:
                print(str(item))

    stime = time.time()

    # securing all error points
    err_nodes, err_exceps = list(), list()
    for k, v in initial_errors.items():
        error_entities, root, catastro = v
        if not catastro:
            for error_entity in error_entities:
                err_nodes.append(error_entity.var_name)
                err_exceps.append(error_entity)

    success = False

    precond_module = PrecondGenModule(model)
    # I only need the zero_grad method from an optimizer, therefore any optimizer works
    optimizer = torch.optim.Adam(precond_module.parameters(), lr=0.1)

    for kk, vv in precond_module.abstracts.items():
        print(kk, 'lb     :', vv.lb)
        print(kk, 'ub     :', vv.ub)

    for iter in range(100):
        print('----------------')
        optimizer.zero_grad()
        loss, errors = precond_module.forward(err_nodes, err_exceps)
        # for kk, vv in precond_module.abstracts.items():
        #     try:
        #         vv.lb.retain_grad()
        #         vv.ub.retain_grad()
        #     except:
        #         print(kk, 'cannot retain grad')

        print('iter =', iter, 'loss =', loss, '# errors =', len(errors))

        if len(errors) == 0:
            success = True
            print('securing condition found!')
            break

        loss.backward()

        # for kk, vv in precond_module.abstracts.items():
        #     print(kk, 'lb grad:', vv.lb.grad)
        #     print(kk, 'ub grad:', vv.ub.grad)
        #     print(kk, 'lb     :', vv.lb)
        #     print(kk, 'ub     :', vv.ub)

        precond_module.grad_step()

    # clip by initial abstracts
    range_clipping(precond_module.model.initial_abstracts, precond_module.centers, precond_module.scales, precond_module.spans)

    for kk, vv in precond_module.abstracts.items():
        print(kk, 'lb     :', vv.lb)
        print(kk, 'ub     :', vv.ub)
    #     print(kk, 'lb grad:', vv.lb.grad)
    #     print(kk, 'ub grad:', vv.ub.grad)

    print('--------------')
    if success:
        print('Success!')
        precond_module.precondition_study()
    else:
        print('!!! Not success')
        raise Exception('failed here :(')
    print(f'Time elapsed: {time.time() - stime:.3f} s')
    print('--------------')


# securing individual error point
# for k, v in initial_errors.items():
#     error_entities, root, catastro = v
#     if not catastro:
#         for error_entity in error_entities:
#             print(f'now working on {error_entity.var_name}')
#             precond_module = PrecondGenModule(model)
#
#             success = False
#
#             # I only need the zero_grad method from an optimizer, therefore any optimizer works
#             optimizer = torch.optim.Adam(precond_module.parameters(), lr=0.1)
#
#
#             for kk, vv in precond_module.abstracts.items():
#                 print(kk, 'lb     :', vv.lb)
#                 print(kk, 'ub     :', vv.ub)
#
#             for iter in range(100):
#                 print('----------------')
#                 optimizer.zero_grad()
#                 loss, errors = precond_module.forward(error_entity.var_name, error_entity)
#                 # for kk, vv in precond_module.abstracts.items():
#                 #     try:
#                 #         vv.lb.retain_grad()
#                 #         vv.ub.retain_grad()
#                 #     except:
#                 #         print(kk, 'cannot retain grad')
#
#                 print('iter =', iter, 'loss =', loss, '# errors =', len(errors))
#
#                 loss.backward()
#
#                 # for kk, vv in precond_module.abstracts.items():
#                 #     print(kk, 'lb grad:', vv.lb.grad)
#                 #     print(kk, 'ub grad:', vv.ub.grad)
#                 #     print(kk, 'lb     :', vv.lb)
#                 #     print(kk, 'ub     :', vv.ub)
#
#                 precond_module.grad_step()
#                 if k not in errors:
#                     success = True
#                     print('securing condition found!')
#                     break
#
#             for kk, vv in precond_module.abstracts.items():
#                 print(kk, 'lb     :', vv.lb)
#                 print(kk, 'ub     :', vv.ub)
#             #     print(kk, 'lb grad:', vv.lb.grad)
#             #     print(kk, 'ub grad:', vv.ub.grad)
#
#             print('--------------')
#             if success: print('Success!')
#             else:
#                 print('!!! Not success')
#                 raise Exception('failed here :(')
#             print(f'Time elapsed: {time.time() - stime:.3f} s')
#             print('--------------')
