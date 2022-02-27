"""
    This script tries to generate preconditon that secures the model from numerical bugs via gradient descent.
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
from interp.interp_module import load_onnx_from_file
from interp.interp_utils import PossibleNumericalError, parse_attribute, discrete_types
from interp.specified_vars import nospan_vars
from interp.specified_vars import nodiff_vars as nodiff_vars_default
from trigger.hints import hard_coded_input_output_hints, customized_lr_inference_inst_gen


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

class InducingInputGenModule(nn.Module):
    """
        The class for generating error-inducing inputs and weights with interval span via gradient descent.
    """

    def __init__(self, model, seed=42, span_len=1e-3, no_diff_vars=list(), no_span_vars=list(), from_init_dict=False):
        super(InducingInputGenModule, self).__init__()
        # assure that the model is already analyzed
        assert model.abstracts is not None

        new_initial_abstracts = dict()

        self.seed = seed
        self.span_len = span_len
        self.from_init_dict = from_init_dict

        # centers and scales are parameters
        self.centers = dict()
        self.scales = dict()
        self.spans = dict()

        # backup initial centers
        self.initial_centers = dict()
        self.expand_initial_spans = dict()

        self.model = model
        self.abstracts = dict()

        self.orig_values = dict()

        for s, orig in model.initial_abstracts.items():
            new_abst = self.construct(orig, s, s in no_diff_vars, s in no_span_vars, self.model.signature_dict[s][0], self.from_init_dict)
            self.abstracts[s] = new_abst

        for k, v in self.centers.items():
            self.initial_centers[k] = v.detach().clone()
            # print(f'initial abstract for {k}')
            # model.initial_abstracts[k].print()
            # print(f'initial grounded: {k} = {self.initial_centers[k]}')

        no = 0
        for k, v in self.centers.items():
            self.register_parameter(f'centers:{no}', v)
            no += 1

    def rewind(self):
        for k in self.initial_centers:
            self.centers[k].data = self.initial_centers[k].detach().clone()

    def set_span_len(self, new_span_len, no_span_vars):
        self.span_len = new_span_len
        for var_name in self.scales:
            if self.model.signature_dict[var_name][0] in discrete_types or var_name in no_span_vars:
                pass
            else:
                self.scales[var_name].fill_(new_span_len)

    def clip_to_valid_range(self, vanilla_lb_ub):
        for var_name in self.centers:
            if var_name in vanilla_lb_ub:
                if var_name in self.spans:
                    now_lb, now_ub = vanilla_lb_ub[var_name]
                    now_lb += torch.max(self.spans[var_name] * self.scales[var_name]).item()
                    now_ub -= torch.max(self.spans[var_name] * self.scales[var_name]).item()
                else:
                    now_lb, now_ub = vanilla_lb_ub[var_name]
                self.centers[var_name].data.clamp_(min=now_lb, max=now_ub)

    def forward(self, node, excep, div_branch='+'):
        """
            construct robust error instance generation loss
        :param node: the node name (str) or (list of that) that causes numerical error
        :param excep: the numerical error object or (list of that)
        :param div_branch: for div op, whether to use positive branch or negative branch, '+' or '-'
        :return:
        """
        self.abstracts.clear()
        for s, orig in self.model.initial_abstracts.items():
            new_abst = self.construct(orig, s, self.from_init_dict)
            self.abstracts[s] = new_abst
        _, errors = self.model.forward(self.abstracts, {'diff_order': 1, 'concrete_rand': self.seed})

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
            prev_node = prev_nodes[0][0]
            if excep.optype == 'Log' or excep.optype == 'Sqrt':
                # print(node)
                prev_abs = self.abstracts[prev_node]
                # loss += torch.max(prev_abs.ub * (prev_abs.lb < PossibleNumericalError.UNDERFLOW_LIMIT))
                prev_prev_nodes = self.find_prev_nodes_optypes(prev_node)
                if len(prev_prev_nodes) > 0 and prev_prev_nodes[0][1] == 'Softmax' and self.abstracts[prev_prev_nodes[0][0]].lb.numel() > 1:
                    # use prev prev node
                    # loss penetration through softmax due to optimization challenge
                    prev_prev_abs = self.abstracts[prev_prev_nodes[0][0]]
                    # prev_prev_abs.print()

                    # v3
                    loss += torch.sum(prev_prev_abs.ub.min(axis=-1)[0]) - torch.sum(prev_prev_abs.lb) + \
                    torch.sum(torch.gather(prev_prev_abs.lb, dim=-1, index=torch.argmin(prev_prev_abs.ub, dim=-1, keepdim=True)))

                    # v4
                    # axis = parse_attribute(prev_nodes[0][2]).get('axis', -1)
                    # loss += torch.sum(- torch.max(prev_abs.lb, dim=axis)[0] + torch.min(prev_abs.ub, dim=axis)[0])
                else:
                    loss += torch.min(prev_abs.ub.view(-1), dim=0)[0]
            elif excep.optype == 'Exp':
                prev_abs = self.abstracts[prev_node]
                thres = PossibleNumericalError.OVERFLOW_D * np.log(10)
                loss += torch.sum(torch.maximum(- prev_abs.lb + thres, torch.zeros_like(prev_abs.lb)))
            elif excep.optype == 'LogSoftMax':
                axis = parse_attribute(prev_nodes[0][2]).get('axis', -1)
                prev_abs = self.abstracts[prev_node]
                loss += torch.sum(- torch.max(prev_abs.lb, dim=axis)[0] + torch.min(prev_abs.ub, dim=axis)[0] - (PossibleNumericalError.OVERFLOW_D - PossibleNumericalError.UNDERFLOW_D))
            elif excep.optype == 'Div':
                # contains zero in div
                divisor = [item for item in prev_nodes if item[3] == 1][0]
                divisor_abs = self.abstracts[divisor[0]]
                loss += torch.sum((torch.abs(divisor_abs.lb) + torch.abs(divisor_abs.ub)))
            else:
                raise Exception(f'excep type {excep.optype} not supported yet, you can follow above template to extend the support')
        return loss, errors

    def robust_error_check(self, node, excep):
        # self.abstracts['MatMul:0'].print()
        err_nodes = list()
        robust_exceps = list()

        if not isinstance(node, list):
            nodes, exceps = [node], [excep]
        else:
            nodes, exceps = node, excep

        for node, excep in zip(nodes, exceps):
            prev_nodes = self.find_prev_nodes_optypes(node)
            prev_node = prev_nodes[0][0]
            if excep.optype == 'Log' or excep.optype == 'Sqrt':
                if prev_node in self.abstracts:
                    prev_abs = self.abstracts[prev_node]
                    if torch.any(prev_abs.ub <= PossibleNumericalError.UNDERFLOW_LIMIT):
                        err_nodes.append(node)
                        robust_exceps.append(excep)
                else:
                    print(f'! cannot find the abstraction for the previous node of {node}')
            elif excep.optype == 'Exp':
                if prev_node in self.abstracts:
                    prev_abs = self.abstracts[prev_node]
                    if torch.any(prev_abs.lb > PossibleNumericalError.OVERFLOW_D * np.log(10)):
                        err_nodes.append(node)
                        robust_exceps.append(excep)
                else:
                    print(f'! cannot find the abstraction for the previous node of {node}')
            elif excep.optype == 'LogSoftMax':
                axis = parse_attribute(prev_nodes[0][2]).get('axis', -1)
                if prev_node in self.abstracts:
                    prev_abs = self.abstracts[prev_node]
                    if torch.any(torch.max(prev_abs.lb, dim=axis)[0] - torch.min(prev_abs.ub, dim=axis)[0] >= (PossibleNumericalError.OVERFLOW_D - PossibleNumericalError.UNDERFLOW_D)):
                        err_nodes.append(node)
                        robust_exceps.append(excep)
            elif excep.optype == 'Div':
                divisor = [item for item in prev_nodes if item[3] == 1][0]
                if divisor[0] in self.abstracts:
                    divisor_abs = self.abstracts[divisor[0]]
                    if torch.any((divisor_abs.lb >= -PossibleNumericalError.UNDERFLOW_LIMIT) & (divisor_abs.ub <= PossibleNumericalError.UNDERFLOW_LIMIT)):
                        err_nodes.append(node)
                        robust_exceps.append(excep)

        return err_nodes, robust_exceps


    def robust_input_study(self, print_stdout=True):
        """
            summarize the heuristics of generated preconditions
        :param print: whether to print to console
        :return:
        """
        tot_nodes = len(self.initial_centers)
        changed_nodes = 0
        unchanged_nodes = 0
        inf_change_nodes = 0
        average_abs_drift = list()
        average_relative_drift = list()
        maximum_abs_drift = 0.
        minimum_abs_drift = 1e+20
        maximum_relative_drift = 0.
        minimum_relative_drift = 1e+20
        for s in self.initial_centers:
            if self.tensor_equal(self.initial_centers[s], self.centers[s]):
                unchanged_nodes += 1
            else:
                changed_nodes += 1
                now_abs_drift = torch.mean(torch.abs(self.initial_centers[s] - self.centers[s])).detach().cpu().item()
                now_relative_drift = torch.mean(torch.abs(self.initial_centers[s] - self.centers[s]) / (self.expand_initial_spans[s])).detach().cpu().item()
                average_abs_drift.append(now_abs_drift)
                maximum_abs_drift = max(maximum_abs_drift, now_abs_drift)
                minimum_abs_drift = min(minimum_abs_drift, now_abs_drift)
                if not math.isfinite(now_relative_drift):
                    inf_change_nodes += 1
                else:
                    average_relative_drift.append(now_relative_drift)
                    maximum_relative_drift = max(maximum_relative_drift, now_relative_drift)
                    minimum_relative_drift = min(minimum_relative_drift, now_relative_drift)

        ans = {
            'span_len': self.span_len,
            'tot_nodes': tot_nodes,
            'tot_changed_nodes': changed_nodes,
            'tot_unchanged_nodes': unchanged_nodes,
            'tot_infchanged_nodes': inf_change_nodes,
            'average_abs_drift': np.mean(average_abs_drift),
            'average_relative_drift': np.mean(average_relative_drift),
            'maximum_abs_drift': maximum_abs_drift,
            'minimum_abs_drift': minimum_abs_drift,
            'maximum_relative_drift': maximum_relative_drift,
            'minimum_relative_drift': minimum_relative_drift
        }
        if print_stdout:
            print(json.dumps(ans, indent=2))
        return ans

    def dump_init_weights(self):
        # indeed it is not limited to weights, but also input nodes
        return self.initial_centers

    def dump_gen_weights(self):
        # indeed it is not limited to weights, but also input nodes
        return self.centers

    def construct(self, original: Abstraction, var_name: str, no_diff=False, no_span=False, node_dtype='FLOAT', from_init_dict=False):
        """
            From the original abstraction to create a new abstraction parameterized by span * scale and center:
            [center - span * scale, center + span * scale]
            where center and scale are learnable parameters
        :param original: original abstraction from the first run
        :return: parameterized abstraction
        """
        def _work(lb, ub, shape, splits, name, center_spec=None):

            expand_lb = expand(lb, shape, splits)
            expand_ub = expand(ub, shape, splits)
            torch.manual_seed(self.seed)
            if center_spec is None:
                init_data = torch.rand_like(expand_lb, device=expand_lb.device)
            else:
                # print(f'init {name} from initializer dict')
                # print(center_spec.shape)
                # print(expand_lb.shape)
                init_data = torch.tensor(center_spec, dtype=expand_lb.dtype, device=expand_lb.device)
            self.expand_initial_spans[name] = expand_ub - expand_lb

            if node_dtype in discrete_types:
                if center_spec is None:
                    init_data = torch.rand_like(expand_lb, device=expand_lb.device)
                    init_data = (expand_lb + init_data * (expand_ub - expand_lb + 1)).floor()
                else:
                    init_data = torch.tensor(center_spec, dtype=expand_lb.dtype, device=expand_lb.device)
            else:
                if center_spec is None:
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
                    center_spec = self.model.initializer_dict[var_name][1][i] if from_init_dict and var_name in self.model.initializer_dict else None
                    item_lb, item_ub = _work(original.lb[i], original.ub[i], original.shape[i], original.splits[i], var_name + f'_{i}', center_spec)
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
                center_spec = self.model.initializer_dict[var_name][1] if from_init_dict and var_name in self.model.initializer_dict else None
                ans.lb, ans.ub = _work(original.lb, original.ub, original.shape, original.splits, var_name, center_spec)
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
            return all([InducingInputGenModule.tensor_equal(item_a, item_b) for item_a, item_b in zip(a, b)])
        else:
            return torch.norm(a.view(-1) - b.view(-1), p=float('inf')).detach().cpu().item() <= EPS


stime = time.time()
def prompt(msg):
    print(f'[{time.time() - stime:.3f}s] ' + msg)

def inducing_inference_inst_gen(modelpath, mode, seed, dumping_folder=None, default_lr=1., default_lr_decay=0.1, default_step=500,
                                max_iters=100, span_len_step=np.sqrt(10), span_len_start=1e-10/2.0,
                                skip_stages=list()):
    """
        The wrapped function for inference time input generation
    :param modelpath:
    :param mode: 'rand-input-rand-weight' / 'spec-input-rand-weight' / 'spec-input-spec-weight' / 'all'
    :return:
    """

    model = load_onnx_from_file(modelpath,
                                customize_shape={'unk__766': 572, 'unk__767': 572, 'unk__763': 572, 'unk__764': 572})
    prompt('model initialized')

    bare_name = modelpath.split('/')[-1].split('.')[0]

    initial_errors = model.analyze(model.gen_abstraction_heuristics(os.path.split(modelpath)[-1].split('.')[0]), {'average_pool_mode': 'coarse', 'diff_order': 1})

    if len(initial_errors) == 0:
        print('No numerical bug')
    else:
        print(f'{len(initial_errors)} possible numerical bug(s)')
        for k, v in initial_errors.items():
            print(f'- On tensor {k} triggered by operator {v[1]}:')
            for item in v[0]:
                print(str(item))

    stats_dict = dict()

    # store original input abstraction for clipping
    vanilla_lb_ub = dict()
    for s, abst in model.initial_abstracts.items():
        vanilla_lb_ub[s] = (torch.min(abst.lb).item(), torch.max(abst.ub).item())
    # print(vanilla_lb_ub)

    # scan all error points
    err_nodes, err_exceps = list(), list()
    for k, v in initial_errors.items():
        error_entities, root, catastro = v
        if not catastro:
            for error_entity in error_entities:
                err_nodes.append(error_entity.var_name)
                err_exceps.append(error_entity)

    if not os.path.exists(os.path.join(dumping_folder, bare_name)):
        os.makedirs(os.path.join(dumping_folder, bare_name))

    err_seq_no = 0
    for err_node, err_excep in zip(err_nodes, err_exceps):
        stime = time.time()

        now_stats_item = dict()
        success = False
        category = None

        if (mode == 'rand-input-rand-weight' or mode == 'all') and 'rand-input-rand-weight' not in skip_stages:
            print('rand-input-rand-weight stage')
            ls_time = time.time()
            inputgen_module = InducingInputGenModule(model, seed=seed, no_diff_vars=nodiff_vars_default, no_span_vars=nospan_vars,
                                                     from_init_dict=True,
                                                     span_len=0.)
            loss, errors = inputgen_module.forward([err_node], [err_excep])
            trigger_nodes, robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)
            if err_node in trigger_nodes:
                # means random input and random weights can trigger the error
                success = True
                category = 'rand-input-rand-weight'

                # try to maximize span length
                span_len = 0.
                try_span_len = span_len_start
                while try_span_len < 0.5:
                    print('try span len =', try_span_len)
                    inputgen_module.set_span_len(span_len, no_span_vars=nospan_vars)
                    inputgen_module.clip_to_valid_range(vanilla_lb_ub)
                    loss, errors = inputgen_module.forward([err_node], [err_excep])
                    trigger_nodes, robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)
                    if err_node not in trigger_nodes:
                        break
                    else:
                        span_len = try_span_len
                        try_span_len *= span_len_step
                now_stats_item['span_len'] = span_len

                if dumping_folder is not None:
                    if not os.path.exists(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))):
                        os.makedirs(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt')))
                    torch.save(inputgen_module.dump_init_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))
                    torch.save(inputgen_module.dump_init_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_gen.pt'))

            lt_time = time.time()
            now_stats_item['rand-input-rand-weight-time'] = lt_time - ls_time

        if (mode == 'spec-input-rand-weight' or (mode == 'all' and not success)) and 'spec-input-rand-weight' not in skip_stages:
            print('spec-input-rand-weight stage')
            # detect input nodes, then other nodes are weights and they will be fixed
            ls_time = time.time()
            iters_log = dict()

            # extract model name from path
            if bare_name in hard_coded_input_output_hints:
                input_nodes, loss_function_nodes = hard_coded_input_output_hints[bare_name][0].copy(), hard_coded_input_output_hints[bare_name][1].copy()
            else:
                input_nodes, loss_function_nodes = model.detect_input_and_output_nodes(alert_exceps=False)
            no_diff_vars = nodiff_vars_default + [s for s in model.initial_abstracts if s not in input_nodes]


            inputgen_module = InducingInputGenModule(model, seed=seed, no_diff_vars=no_diff_vars, no_span_vars=nospan_vars,
                                                     from_init_dict=True,
                                                     span_len=0.)

            # try to maximize span length
            span_len = -1.
            try_span_len = 0.

            while try_span_len < 0.5:
                print('try span len =', try_span_len)
                now_span_len_success = False

                inputgen_module.set_span_len(try_span_len, no_span_vars=nospan_vars)
                optimizer = torch.optim.Adam(inputgen_module.parameters(), lr=default_lr if bare_name not in customized_lr_inference_inst_gen else customized_lr_inference_inst_gen[bare_name])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=default_step)

                for iter in range(max_iters):
                    # print('----------------')

                    optimizer.zero_grad()
                    loss, errors = inputgen_module.forward([err_node], [err_excep])
                    trigger_nodes, robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)

                    print(f'[{bare_name},err{err_seq_no},{err_node}] iter =', iter, 'loss =', loss, 'has robust error =', len(robust_errors))

                    if err_node in trigger_nodes:
                        now_span_len_success = True
                        print('robust error found!')
                        break
                    try:
                        loss.backward()
                    except:
                        break
                    optimizer.step()
                    scheduler.step()

                    inputgen_module.clip_to_valid_range(vanilla_lb_ub)

                iters_log[try_span_len] = iter

                if now_span_len_success:
                    success = True
                    category = 'spec-input-rand-weight'
                    span_len = try_span_len
                    now_stats_item['span_len'] = span_len
                    if try_span_len == 0.:
                        try_span_len = span_len_start
                    else:
                        try_span_len *= span_len_step

                    if dumping_folder is not None:
                        if not os.path.exists(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))):
                            os.makedirs(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt')))
                        torch.save(inputgen_module.dump_init_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))
                        torch.save(inputgen_module.dump_gen_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_gen.pt'))

                else:
                    break

            lt_time = time.time()
            now_stats_item['spec-input-rand-weight-time'] = lt_time - ls_time
            now_stats_item['spec-input-rand-weight-iters'] = iters_log

        if (mode == 'spec-input-spec-weight' or (mode == 'all' and not success)) and 'spec-input-spec-weight' not in skip_stages:
            print('spec-input-spec-weight stage')

            ls_time = time.time()
            iters_log = dict()

            inputgen_module = InducingInputGenModule(model, seed=seed, no_diff_vars=nodiff_vars_default, no_span_vars=nospan_vars,
                                                     from_init_dict=True,
                                                     span_len=0.)

            # try to maximize span length
            span_len = -1.
            try_span_len = 0.

            while try_span_len < 0.5:
                print('try span len =', try_span_len)
                now_span_len_success = False

                inputgen_module.set_span_len(try_span_len, no_span_vars=nospan_vars)
                optimizer = torch.optim.Adam(inputgen_module.parameters(), lr=default_lr if bare_name not in customized_lr_inference_inst_gen else customized_lr_inference_inst_gen[bare_name])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=default_step)

                for iter in range(max_iters):
                    # print('----------------')

                    optimizer.zero_grad()
                    loss, errors = inputgen_module.forward([err_node], [err_excep])
                    trigger_nodes, robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)

                    print(f'[{bare_name},err{err_seq_no},{err_node}] iter =', iter, 'loss =', loss, 'has robust error =', len(robust_errors))

                    if err_node in trigger_nodes:
                        now_span_len_success = True
                        print('robust error found!')
                        break

                    try:
                        loss.backward()
                    except:
                        break
                    optimizer.step()
                    scheduler.step()

                    inputgen_module.clip_to_valid_range(vanilla_lb_ub)

                iters_log[try_span_len] = iter

                if now_span_len_success:
                    success = True
                    category = 'spec-input-spec-weight'
                    span_len = try_span_len
                    now_stats_item['span_len'] = span_len

                    if try_span_len == 0.:
                        try_span_len = span_len_start
                    else:
                        try_span_len *= span_len_step

                    if dumping_folder is not None:
                        if not os.path.exists(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))):
                            os.makedirs(os.path.dirname(os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt')))
                        torch.save(inputgen_module.dump_init_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_init.pt'))
                        torch.save(inputgen_module.dump_gen_weights(), os.path.join(dumping_folder, bare_name, f'{err_seq_no}_{err_node}_gen.pt'))
                else:
                    break

            lt_time = time.time()
            now_stats_item['spec-input-spec-weight-time'] = lt_time - ls_time
            now_stats_item['spec-input-spec-weight-iters'] = iters_log

        now_stats_item['success'] = success
        now_stats_item['category'] = category
        now_stats_item['time'] = time.time() - stime
        now_stats_item['err_seq_no'] = err_seq_no
        stats_dict[err_node] = now_stats_item

        err_seq_no += 1

    return stats_dict


# ================ # ================
# below is for individual instance running

parser = argparse.ArgumentParser()
parser.add_argument('modelpath', type=str, help='model architecture file path')


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

    inputgen_module = InducingInputGenModule(model, no_diff_vars=nodiff_vars_default, no_span_vars=nospan_vars,
                                             from_init_dict=True, span_len=1e-4)
    optimizer = torch.optim.Adam(inputgen_module.parameters(), lr=1)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500)

    # for kk, vv in inputgen_module.abstracts.items():
    # for kk in ['Variable_2/read:0']:
    #     vv = inputgen_module.abstracts[kk]
    #     print(kk, 'lb     :', vv.lb)
    #     print(kk, 'ub     :', vv.ub)

    for err_node, err_excep in zip(err_nodes, err_exceps):
        for iter in range(1000):
            print('----------------')

            optimizer.zero_grad()
            loss, errors = inputgen_module.forward([err_node], [err_excep])

            # for kk, vv in inputgen_module.abstracts.items():
            #     try:
            #         vv.lb.requires_grad_(True)
            #         vv.ub.requires_grad_(True)
            #         vv.lb.retain_grad()
            #         vv.ub.retain_grad()
            #     except:
            #         print(kk, 'cannot retain grad')

            robust_errors = inputgen_module.robust_error_check(err_nodes, err_exceps)

            print('iter =', iter, 'loss =', loss, '# robust errors =', len(robust_errors), '/', len(err_nodes))

            if len(robust_errors) > 0:
                success = True
                print('robust error found!')
                break

            loss.backward()
            optimizer.step()
            scheduler.step()

            # for kk, vv in inputgen_module.abstracts.items():
            # for kk in ['dense_2/kernel/read:0']:
            #     vv = inputgen_module.abstracts[kk]
            #     # print(kk, 'lb grad:', vv.lb.grad)
            #     # print(kk, 'ub grad:', vv.ub.grad)
            #     print(kk, 'lb     :', vv.lb)
            #     print(kk, 'ub     :', vv.ub)


    # for kk, vv in precond_module.abstracts.items():
    #     print(kk, 'lb     :', vv.lb)
    #     print(kk, 'ub     :', vv.ub)
    #     print(kk, 'lb grad:', vv.lb.grad)
    #     print(kk, 'ub grad:', vv.ub.grad)

        print('--------------')
        if success:
            print('Success!')
            inputgen_module.robust_input_study()
        else:
            print('!!! Not success')
            # raise Exception('failed here :(')
        print(f'Time elapsed: {time.time() - stime:.3f} s')
        print('--------------')




