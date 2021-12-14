"""
    This script tries to generate error-inducing training data
"""

EPS = 1e-5

import time
import argparse
import os
import json
import math

import torch
import torch.nn as nn
import torch.optim

import numpy as np

from interp.interp_operator import Abstraction
from interp.interp_utils import discrete_types, PossibleNumericalError


# key: bare_name, values: input_nodes, loss nodes
hard_coded_hints = {'5': (['0', '1'], ['42']), '15': (['y:0', 'X:0', 'S1:0', 'S2:0'], ['cost:0']), '22': (['v', 'v.1'], ['27']), '62': (['v.1'], ['65']), '63': (['v.1'], ['65'])}

params = {'33': {'diff_order': 0}, '40': {'diff_order': 0}, '41': {'diff_order': 0}, '42': {'diff_order': 0}}


class TrainGenModule(nn.Module):
    """
        The class for generating secure preconditon with gradient descent.
    """

    def __init__(self, model, input_nodes, weight_nodes, loss_node_names, require_weight_diffs, seed=42, no_diff_vars=list(), diff_order=2):
        super(TrainGenModule, self).__init__()
        # assure that the model is already analyzed
        assert model.abstracts is not None

        new_initial_abstracts = dict()

        self.seed = seed

        # inputs are parameters
        self.inputs = dict()

        # backup initial inputs
        self.initial_inputs = dict()

        self.model = model
        self.abstracts = dict()

        for s, orig in model.initial_abstracts.items():
            sig_t, sig_shape = model.signature_dict[s]
            if s in input_nodes:
                init_input = nn.Parameter(input_nodes[s], requires_grad=not s in no_diff_vars and sig_t not in discrete_types)
            else:
                init_input = torch.tensor(weight_nodes[s], requires_grad=not s in no_diff_vars and sig_t not in discrete_types)
            self.inputs[s] = init_input

        for k, v in self.inputs.items():
            self.initial_inputs[k] = v.detach().clone()

        no = 0
        for k, v in self.inputs.items():
            if k in input_nodes:
                self.register_parameter(f'inputs:{no}', v)
                no += 1

        # store loss_nodes and require_weight_diffs
        # self.weight_node_names = list(weight_nodes.keys())
        self.loss_node_names = loss_node_names
        self.require_weight_diffs = require_weight_diffs

        self.diff_order = diff_order

    def rewind(self):
        for k in self.initial_inputs:
            self.inputs[k].data = self.initial_inputs[k].detach().clone()

    def forward(self, learning_rate=1., print_more=False):
        """
            construct gradient leakage attack loss
        :return:
        """
        self.abstracts.clear()

        for s, orig in model.initial_abstracts.items():
            sig_t, sig_shape = model.signature_dict[s]

            now_abst = Abstraction()
            now_abst.lb = self.inputs[s]
            now_abst.ub = self.inputs[s]
            now_abst.shape = sig_shape.copy()
            now_abst.splits = [list(range(dim)) for dim in sig_shape]
            now_abst.var_name = s
            self.abstracts[s] = now_abst

        _, errors = self.model.forward(self.abstracts, {'diff_order': self.diff_order, 'concrete_rand': self.seed, 'continue_prop': True})

        avail_loss_node_tot = 0
        loss_node_tot = len(self.loss_node_names)
        loss_to_diff = torch.tensor(0.)
        for loss_node_name in self.loss_node_names:
            if loss_node_name in self.abstracts:
                # lb and ub should be the same
                loss_to_diff += self.abstracts[loss_node_name].lb.sum()
                avail_loss_node_tot += 1

        if avail_loss_node_tot == 0:
            print(f'caution: unavailable loss due to {len(errors)} bugs found')
            loss = torch.tensor(0.)
            return loss, len(errors)

        list_of_weights = [self.abstracts[s].lb for s in self.require_weight_diffs if self.abstracts[s].lb.requires_grad]
        grads = torch.autograd.grad(loss_to_diff, list_of_weights, create_graph=True, allow_unused=True)

        if print_more:
            i = 0
            for s in self.require_weight_diffs:
                if self.abstracts[s].lb.requires_grad:
                    if s in ['discriminator/D_w3/read:0']:
                        print(s, '  grad =')
                        print(grads[i])
                    i += 1
            # debug
            # for s in ['9', '10']:
            #     print(s, 'value =')
            #     print(self.abstracts[s].lb)

        loss = torch.tensor(0.)
        i = 0
        for s, diff_value in self.require_weight_diffs.items():
            if self.abstracts[s].lb.requires_grad:
                if grads[i] is not None:
                    loss += ((grads[i] - diff_value / float(learning_rate)) ** 2).sum()
                i += 1

        # fix for unstable gradient which can happen for L-BFGS
        if torch.isnan(loss):
            loss = torch.tensor(10e+20)
            return loss, len(errors)

        if print_more:
            for s, v in self.abstracts.items():
                # if s in ['discriminator/MatMul_3:0', 'discriminator/Sigmoid:0', 'Log:0', 'add:0', 'Log_1:0']:
                if s in ['x:0', 'z:0', 'discriminator/MatMul_3:0', 'discriminator/Sigmoid:0', 'Log:0']:
                    print(s, 'value =', v.lb)
            if torch.isnan(loss):
                for s, v in self.abstracts.items():
                    print(s, 'value =', v.lb)
                exit(0)

        return loss, len(errors)

    def check(self, learning_rate=1.):
        self.abstracts.clear()

        for s, orig in model.initial_abstracts.items():
            sig_t, sig_shape = model.signature_dict[s]

            now_abst = Abstraction()
            now_abst.lb = self.inputs[s]
            now_abst.ub = self.inputs[s]
            now_abst.shape = sig_shape.copy()
            now_abst.splits = [list(range(dim)) for dim in sig_shape]
            now_abst.var_name = s
            self.abstracts[s] = now_abst

        _, errors = self.model.forward(self.abstracts, {'diff_order': 0, 'concrete_rand': self.seed, 'continue_prop': True})

        avail_loss_node_tot = 0
        loss_node_tot = len(self.loss_node_names)
        loss_to_diff = torch.tensor(0.)
        for loss_node_name in self.loss_node_names:
            if loss_node_name in self.abstracts:
                # lb and ub should be the same
                loss_to_diff += self.abstracts[loss_node_name].lb.sum()
                avail_loss_node_tot += 1

        if avail_loss_node_tot == 0:
            print(f'caution: unavailable loss due to {len(errors)} bugs found')
            return len(errors)

        list_of_weights = [self.abstracts[s].lb for s in self.require_weight_diffs if self.abstracts[s].lb.requires_grad]
        grads = torch.autograd.grad(loss_to_diff, list_of_weights, create_graph=True, allow_unused=True)

        new_abstracts = dict()

        i = 0
        for s, diff_value in self.require_weight_diffs.items():
            if self.abstracts[s].lb.requires_grad:
                if grads[i] is not None:
                    w_new = self.inputs[s] - learning_rate * grads[i]
                    new_abstracts[s] = w_new.clone()
                i += 1
        # others (mostly input data nodes) are kept the same as initial
        for s in self.initial_inputs:
            if s not in new_abstracts:
                new_abstracts[s] = self.initial_inputs[s].clone()

        for s, t in new_abstracts.items():
            tmp = Abstraction()
            tmp.lb = t
            tmp.ub = t
            tmp.shape = self.abstracts[s].shape
            tmp.splits = self.abstracts[s].splits
            tmp.var_name = self.abstracts[s].var_name
            new_abstracts[s] = tmp

        _, errors = self.model.forward(new_abstracts, {'diff_order': 0, 'concrete_rand': self.seed, 'continue_prop': False})
        return len(errors)



parser = argparse.ArgumentParser()
parser.add_argument('chkpdir', type=str, help='directory that contains the checkpoint files')
parser.add_argument('modelpath', type=str, help='model architecture file path')


stime = time.time()
def prompt(msg):
    print(f'[{time.time() - stime:.3f}s] ' + msg)

from interp.interp_operator import Abstraction
from interp.interp_module import load_onnx_from_file, InterpModule

if __name__ == '__main__':

    args = parser.parse_args()

    model = load_onnx_from_file(args.modelpath,
                                customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
    prompt('model initialized')

    # extract model name from path
    bare_name = args.modelpath.split('/')[-1].split('.')[0]
    print('bare_name:', bare_name)

    # read checkpoint file from the given dir
    # note: special naming conventions for the dir is required: need to exist a folder with bare name, and the init/buggy pts are named by '{error_no}_init.pt'/'{error_no}_gen.pt'
    chkp_dir = os.path.join(args.chkpdir, bare_name)
    possible_pts = [fpath for fpath in os.listdir(chkp_dir) if fpath.endswith('.pt')]
    print(possible_pts)
    tot_err = len(possible_pts) // 2

    # store the original lb and ub of the model so that we can clip the input accordingly later
    model.analyze(model.gen_abstraction_heuristics(args.modelpath.split('.')[0]), {'average_pool_mode': 'coarse'})
    vanilla_lb_ub = dict()
    for s, abst in model.initial_abstracts.items():
        vanilla_lb_ub[s] = (torch.min(abst.lb).item(), torch.max(abst.ub).item())
    model.possible_numerical_errors.clear()

    for now_err_no in range(tot_err):
        print('')
        print(f'now handle error #{now_err_no}')

        init_pt = os.path.join(chkp_dir, f'{now_err_no}_init.pt')
        inference_pt = os.path.join(chkp_dir, f'{now_err_no}_gen.pt')

        # load init weights + inputs and buggy inference weights + inputs
        init_dict = torch.load(init_pt)
        inference_dict = torch.load(inference_pt)

        err_stime = time.time()
        success = False
        mode = ''

        # check whether init weights + inputs == infernece weights + inputs
        # if so, we don't need to generate training inputs at all
        sum_diff = 0.
        sum_diff_cells = 0
        sum_diff_vars = 0
        sum_tot_cells = 0
        sum_tot_vars = len(init_dict)
        has_diff = False
        for k in init_dict.keys():
            now_delta = init_dict[k] - inference_dict[k]
            now_delta = now_delta.view(-1)
            now_delta = torch.linalg.norm(now_delta, ord=1)
            if (now_delta >= 1e-5):
                sum_diff += now_delta
                sum_diff_cells += torch.numel(init_dict[k])
                sum_diff_vars += 1
                has_diff = True
            sum_tot_cells += torch.numel(init_dict[k])
        print('=' * 20)
        print('L1 diff:', sum_diff)
        print('diff cells:', sum_diff_cells, '/', sum_tot_cells)
        print('diff vars:', sum_diff_vars, '/', sum_tot_vars)
        print('=' * 20)

        if not has_diff:
            print('no diff, pass')
            success = True
            mode = 'init weight with init input can already trigger bug'
        else:
            # initialize abstraction as concrete tensors from init
            init_phase_abstracts = dict()
            for s in model.start_points:
                if s == InterpModule.SUPER_START_NODE: continue
                sig_t, sig_shape = model.signature_dict[s]

                # TODO: support list-typed nodes
                try:
                    assert s in init_dict
                except Exception:
                    raise Exception(f'node {s} does not exist in dumped data')

                tensor_v = init_dict[s]

                try:
                    assert (all([x == y for x, y in zip(tensor_v.shape, sig_shape)]))
                except Exception:
                    raise Exception(f'node {s}: dumped data shape({tensor_v.shape}) mismatches specified data shape in signature({sig_shape})')

                now_abst = Abstraction()
                now_abst.lb = torch.clone(tensor_v)
                now_abst.ub = torch.clone(tensor_v)
                now_abst.shape = sig_shape.copy()
                now_abst.splits = [list(range(dim)) for dim in sig_shape]
                now_abst.var_name = s + '_init'
                init_phase_abstracts[s] = now_abst

            # benign forward pass, terminate='' causes it to traverse all nodes
            model.analyze(from_existing_abstract=init_phase_abstracts, terminate='')

            if len(model.possible_numerical_errors) > 0:
                print(f'! warning: there exists possible numerical bug even with the benign input. The abstraction dict may be incomplete -- we may not able to locate real loss node')
            pre_bug_num = len(model.possible_numerical_errors)

            # locate inputs nodes and loss function nodes
            if bare_name in hard_coded_hints:
                input_nodes, loss_function_nodes = hard_coded_hints[bare_name][0].copy(), hard_coded_hints[bare_name][1].copy()
            else:
                input_nodes, loss_function_nodes = model.detect_input_and_output_nodes()

            print('input nodes:', input_nodes)
            print('loss nodes:', loss_function_nodes)

            # check whether weight nodes need change; if not, then all differences between init and buggy settings lie in input nodes
            # in this case, we don't need to generate training inputs to trigger the bug
            sum_diff = 0.
            sum_diff_cells = 0
            sum_diff_vars = 0
            sum_tot_cells = 0
            sum_tot_vars = len(init_dict)
            has_diff = False
            for k in init_dict.keys():
                if k not in input_nodes:
                    now_delta = init_dict[k] - inference_dict[k]
                    now_delta = now_delta.view(-1)
                    now_delta = torch.linalg.norm(now_delta, ord=1)
                    if (now_delta >= 1e-5):
                        sum_diff += now_delta
                        sum_diff_cells += torch.numel(init_dict[k])
                        sum_diff_vars += 1
                        has_diff = True
                    sum_tot_cells += torch.numel(init_dict[k])
            print('=' * 20)
            print('L1 diff:', sum_diff)
            print('diff cells:', sum_diff_cells, '/', sum_tot_cells)
            print('diff vars:', sum_diff_vars, '/', sum_tot_vars)
            print('=' * 20)
            if not has_diff:
                print('no diff in weight nodes, pass')
                success = True
                mode = 'init weight with specific input can already trigger bug'
                continue

            # main iterations: adapted from deep gradient leakage attack
            traingen_module = TrainGenModule(model, dict([(node, init_dict[node]) for node in input_nodes]),
                                             dict([(node, init_dict[node]) for node in init_dict if node not in input_nodes]),
                                             loss_function_nodes,
                                             dict([(node, init_dict[node] - inference_dict[node]) for node in init_dict if node not in input_nodes]),
                                             diff_order=2 if bare_name not in params else params[bare_name]['diff_order'])

            optimizer = torch.optim.LBFGS(traingen_module.parameters())

            for iters in range(300):

                def closure():
                    optimizer.zero_grad()
                    grad_diff, _ = traingen_module.forward()
                    print('iter =', iters, 'loss =', grad_diff)
                    if grad_diff.requires_grad:
                        grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)

                for k, v in traingen_module.inputs.items():
                    if k in input_nodes:
                        v.data.clamp_(min=vanilla_lb_ub[k][0], max=vanilla_lb_ub[k][1])

                # within valid input range, with init weight, generate bugs by specific training inputs
                # which also means that specific inference input along with init weights can trigger the numerical bug
                # thus, we view this as a success
                fin_loss, num_bug = traingen_module.forward(print_more=False)
                print('iter =', iters, 'final loss =', fin_loss, 'num bug =', num_bug)
                # for k, v in traingen_module.inputs.items():
                #     if k in input_nodes:
                #         print(k, v)

                if num_bug > pre_bug_num:
                    success = True
                    mode = 'found init weight with specific inputs can generate bug'
                    break

                if not success:
                    num_errors = traingen_module.check()
                    success = num_errors > pre_bug_num
                    mode = f'now # errors = {num_errors}, pre # errors = {pre_bug_num}'
                    if success:
                        break

        print('success?', success)
        print('more info:', mode)
