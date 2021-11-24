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
from interp.interp_module import load_onnx_from_file
from interp.interp_utils import PossibleNumericalError, parse_attribute, discrete_types
from interp.specified_vars import nodiff_vars, nospan_vars


def expand(orig_tensor: torch.Tensor, shape: list, splits: list) -> torch.Tensor:
    """
        Expand a original abstract tensor into a concrete tensor by tiling
    :param orig_tensor:
    :param shape:
    :param splits:
    :return: torch.Tensor
    """
    if len(shape) == 0:
        # scalar tensor, return as it is
        return orig_tensor
    else:
        ans_tensor = orig_tensor
        for now_dim in range(len(shape)):
            index = [i
                     for i, item in enumerate(splits[now_dim])
                     for j in range(splits[now_dim][i],
                                    shape[now_dim] if i == len(splits[now_dim]) - 1 else splits[now_dim][i+1])
                     ]
            index = torch.tensor(index, device=ans_tensor.device)
            ans_tensor = torch.index_select(ans_tensor, dim=now_dim, index=index)
        return ans_tensor



parser = argparse.ArgumentParser()
parser.add_argument('modelpath', type=str, help='model architecture file path')


stime = time.time()
def prompt(msg):
    print(f'[{time.time() - stime:.3f}s] ' + msg)


class InducingInputGenModule(nn.Module):
    """
        The class for generating secure preconditon with gradient descent.
    """

    def __init__(self, model, seed=42, span_len=0.001, no_diff_vars=list(), no_span_vars=list()):
        super(InducingInputGenModule, self).__init__()
        # assure that the model is already analyzed
        assert model.abstracts is not None

        new_initial_abstracts = dict()

        self.seed = seed
        self.span_len = span_len

        # centers and scales are parameters
        self.centers = dict()
        self.scales = dict()
        self.spans = dict()

        self.model = model
        self.abstracts = dict()

        self.orig_values = dict()

        for s, orig in model.initial_abstracts.items():
            new_abst = self.construct(orig, s, s in no_diff_vars, s in no_span_vars, self.model.signature_dict[s][0])
            self.abstracts[s] = new_abst

        # backup initial centers
        self.initial_centers = dict()
        for k, v in self.centers.items():
            self.initial_centers[k] = v.detach()
            # print(f'initial abstract for {k}')
            # model.initial_abstracts[k].print()
            print(f'initial grounded: {k} = {self.initial_centers[k]}')

        no = 0
        for k, v in self.centers.items():
            self.register_parameter(f'centers:{no}', v)
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
        _, errors = self.model.forward(self.abstracts, {'diff_order': 1, 'concrete_rand': 42})

        # for k, v in self.abstracts.items():
        #     print(k)
        #     v.print()

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
                # loss += torch.max(prev_abs.ub * (prev_abs.lb < PossibleNumericalError.UNDERFLOW_LIMIT))
                loss += torch.amin(prev_abs.ub)
            elif excep.optype == 'Exp':
                prev_node = prev_nodes[0][0]
                prev_abs = self.abstracts[prev_node]
                thres = PossibleNumericalError.OVERFLOW_D * np.log(10)
                loss += torch.amin(- prev_abs.lb + thres)
            elif excep.optype == 'LogSoftMax':
                prev_node = prev_nodes[0][0]
                axis = parse_attribute(prev_nodes[0][2]).get('axis', -1)
                prev_abs = self.abstracts[prev_node]
                loss += - torch.amax(prev_abs.lb, dim=axis) + torch.amin(prev_abs.ub, dim=axis) - (PossibleNumericalError.OVERFLOW_D - PossibleNumericalError.UNDERFLOW_D)
            elif excep.optype == 'Div':
                # contains zero in div
                divisor = [item for item in prev_nodes if item[3] == 1][0]
                divisor_abs = self.abstracts[divisor[0]]
                loss += torch.abs(divisor_abs.lb) + torch.abs(divisor_abs.ub)
            else:
                raise Exception(f'excep type {excep.optype} not supported yet, you can follow above template to extend the support')
        return loss, errors

    def robust_error_check(self, node, excep):
        robust_exceps = list()

        if not isinstance(node, list):
            nodes, exceps = [node], [excep]
        else:
            nodes, exceps = node, excep

        for node, excep in zip(nodes, exceps):
            prev_nodes = self.find_prev_nodes_optypes(node)
            if excep.optype == 'Log' or excep.optype == 'Sqrt':
                prev_node = prev_nodes[0][0]
                if prev_node in self.abstracts:
                    prev_abs = self.abstracts[prev_node]
                    if torch.any(prev_abs.ub <= PossibleNumericalError.UNDERFLOW_LIMIT):
                        robust_exceps.append(excep)
                else:
                    print(f'! cannot find the abstraction for the previous node of {node}')
            elif excep.optype == 'Exp':
                prev_node = prev_nodes[0][0]
                if prev_node in self.abstracts:
                    prev_abs = self.abstracts[prev_node]
                    if torch.any(prev_abs.lb > PossibleNumericalError.OVERFLOW_D * np.log(10)):
                        robust_exceps.append(excep)
                else:
                    print(f'! cannot find the abstraction for the previous node of {node}')
            elif excep.optype == 'LogSoftMax':
                prev_node = prev_nodes[0][0]
                axis = parse_attribute(prev_nodes[0][2]).get('axis', -1)
                if prev_node in self.abstracts:
                    prev_abs = self.abstracts[prev_node]
                    if torch.any(torch.max(prev_abs.lb, dim=axis) - torch.min(prev_abs.ub, dim=axis) >= (PossibleNumericalError.OVERFLOW_D - PossibleNumericalError.UNDERFLOW_D)):
                        robust_exceps.append(excep)
            elif excep.optype == 'Div':
                divisor = [item for item in prev_nodes if item[3] == 1][0]
                if divisor[0] in self.abstracts:
                    divisor_abs = self.abstracts[divisor[0]]
                    if torch.any((divisor_abs.lb >= -PossibleNumericalError.UNDERFLOW_LIMIT) & (divisor_abs.ub <= PossibleNumericalError.UNDERFLOW_LIMIT)):
                        robust_exceps.append(excep)

        return robust_exceps


    def err_input_study(self, print_stdout=True):
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

    def construct(self, original: Abstraction, var_name: str, no_diff=False, no_span=False, node_dtype='FLOAT'):
        """
            From the original abstraction to create a new abstraction parameterized by span * scale and center:
            [center - span * scale, center + span * scale]
            where center and scale are learnable parameters
        :param original: original abstraction from the first run
        :return: parameterized abstraction
        """
        def _work(lb, ub, shape, splits, name):

            expand_lb = expand(lb, shape, splits)
            expand_ub = expand(ub, shape, splits)
            init_data = torch.rand_like(expand_lb, device=expand_lb.device)

            if node_dtype in discrete_types:
                init_data = torch.rand_like(expand_lb, device=expand_lb.device)
                init_data = (expand_lb + init_data * (expand_ub - expand_lb + 1)).floor()
            else:
                init_data = expand_lb + init_data * (expand_ub - expand_lb)
            init_data = nn.Parameter(init_data, requires_grad=not no_diff and node_dtype not in discrete_types)

            span = torch.tensor(expand_ub - expand_lb, device=expand_ub.device)
            if node_dtype in discrete_types or no_span:
                scale = torch.zeros_like(span)
            else:
                scale = torch.ones_like(span) * self.span_len

            new_lb = init_data - scale * span
            new_ub = init_data + scale * span

            self.centers[var_name] = init_data
            self.spans[var_name] = span
            self.scales[var_name] = scale

            return new_lb, new_ub

        ans = Abstraction()
        splits = None
        if isinstance(original.lb, list):
            ans.lb = list()
            ans.ub = list()
            splits = list()
            for i in range(len(original.lb)):
                if var_name + f'_{i}' in self.centers and var_name + f'_{i}' in self.spans:
                    item_lb = self.centers[var_name + f'_{i}'] - self.spans[var_name + f'_{i}'] * self.scales[var_name + f'_{i}']
                    item_ub = self.centers[var_name + f'_{i}'] + self.spans[var_name + f'_{i}'] * self.scales[var_name + f'_{i}']
                elif var_name + f'_{i}' in self.centers and var_name + f'_{i}' not in self.spans:
                    item_lb = self.centers[var_name + f'_{i}']
                    item_ub = self.centers[var_name + f'_{i}']
                else:
                    item_lb, item_ub = _work(original.lb[i], original.ub[i], original.shape[i], original.splits[i], var_name + f'_{i}')
                ans.lb.append(item_lb)
                ans.ub.append(item_ub)
                splits.append([list(range(shapei)) for shapei in original.shape[i]])
        else:
            if var_name in self.centers and var_name in self.spans:
                ans.lb = self.centers[var_name] - self.spans[var_name] * self.scales[var_name]
                ans.ub = self.centers[var_name] + self.spans[var_name] * self.scales[var_name]
            elif var_name in self.centers and var_name not in self.spans:
                ans.lb = self.centers[var_name]
                ans.ub = self.centers[var_name]
            else:
                ans.lb, ans.ub = _work(original.lb, original.ub, original.shape, original.splits, var_name)
            splits = [list(range(shapei)) for shapei in original.shape]
        ans.shape = original.shape.copy()
        ans.splits = splits
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

    inputgen_module = InducingInputGenModule(model, no_diff_vars=nodiff_vars, no_span_vars=nospan_vars)
    optimizer = torch.optim.Adam(inputgen_module.parameters(), lr=0.1)

    # for kk, vv in precond_module.abstracts.items():
    #     print(kk, 'lb     :', vv.lb)
    #     print(kk, 'ub     :', vv.ub)

    for iter in range(100):
        print('----------------')
        optimizer.zero_grad()
        loss, errors = inputgen_module.forward(err_nodes, err_exceps)

        robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)

        # for kk, vv in inputgen_module.abstracts.items():
        #     try:
        #         vv.lb.retain_grad()
        #         vv.ub.retain_grad()
        #     except:
        #         print(kk, 'cannot retain grad')

        print('iter =', iter, 'loss =', loss, '# robust errors =', len(robust_errors), '/', len(err_nodes))

        loss.backward()
        optimizer.step()

        # for kk, vv in inputgen_module.abstracts.items():
        # for kk in ['add_5:0', 'Softmax:0']:
        #     vv = inputgen_module.abstracts[kk]
        #     print(kk, 'lb grad:', vv.lb.grad)
        #     print(kk, 'ub grad:', vv.ub.grad)
        #     print(kk, 'lb     :', vv.lb)
        #     print(kk, 'ub     :', vv.ub)


        if len(robust_errors) > 0:
            success = True
            print('securing condition found!')
            if len(robust_errors) == len(err_nodes):
                print('triggered all errors')
                break

    # for kk, vv in precond_module.abstracts.items():
    #     print(kk, 'lb     :', vv.lb)
    #     print(kk, 'ub     :', vv.ub)
    #     print(kk, 'lb grad:', vv.lb.grad)
    #     print(kk, 'ub grad:', vv.ub.grad)

    print('--------------')
    if success:
        print('Success!')
        # inputgen_module.precondition_study()
    else:
        print('!!! Not success')
        raise Exception('failed here :(')
    print(f'Time elapsed: {time.time() - stime:.3f} s')
    print('--------------')




