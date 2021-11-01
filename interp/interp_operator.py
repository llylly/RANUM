import math
import numpy as np
import torch
import bisect
from functools import reduce

import onnx
from onnx import numpy_helper
from interp.interp_utils import AbstractionInitConfig, parse_attribute, unsupported_types, datatype_mapping, get_numel, \
    PossibleNumericalError, index_clip_thres


class Abstraction(object):
    """
        The Abstraction class is attached to each variable during the interpretation process
    """

    def __init__(self):
        """
            Vacant initializer, please use load() immediately to construct a legal Abstraction
        """
        self.lb = self.ub = self.var_name = self.shape = self.splits = None

    def load(self,
             config: AbstractionInitConfig,
             var_name: str,
             tensor_shape: tuple or list,
             tensor_type: str,
             tensor_data: None or np.ndarray or list,
             cuda=False):
        """
            Load the abstraction
        :param config:
        :param var_name:
        :param tensor_shape:
        :param tensor_type:
        :param tensor_data:
        :return:
        """

        # # === DEBUG begin ===
        # print('config:', config.to_dict())
        # print('var name:', var_name)
        # print('shape:', tensor_shape)
        # print('type:', tensor_type)
        # if tensor_data is not None:
        #     print('data shape:', tensor_data.shape)
        # print('')
        # # === DEBUG end ===

        diff = config.diff
        lb, ub = config.lb, config.ub
        from_init = config.from_init
        from_init_margin = config.from_init_margin
        stride = config.stride

        if from_init and isinstance(tensor_data, list):
            singleton_abstracts = [
                Abstraction().load(config, var_name, single_data.shape, tensor_type, single_data, cuda) for single_data
                in tensor_data]
            self.lb = [s.lb for s in singleton_abstracts]
            self.ub = [s.ub for s in singleton_abstracts]
            self.shape = [s.shape for s in singleton_abstracts]
            self.splits = [s.splits for s in singleton_abstracts]
            self.var_name = var_name
            return self

        if tensor_type in unsupported_types:
            # if the tensor type is not supported, just create null tensors as placeholders
            self.lb = torch.tensor(data=[])
            self.ub = torch.tensor(data=[])
            self.splits = [[]]
            self.shape = [0]
        else:
            # support legal types
            tensor_shape = list(tensor_shape)
            if len(tensor_shape) == 0:
                stride = 1
            else:
                if isinstance(stride, int):
                    stride = [stride for _ in tensor_shape]
                try:
                    assert len(stride) == len(tensor_shape)
                except:
                    raise Exception(f'Variable {var_name}: stride config {stride} should match {tensor_shape}')

            if len(tensor_shape) == 0:
                # scalar
                if from_init and tensor_data is not None:
                    tensor_data = tensor_data.reshape(())
                    self.lb, self.ub = \
                        torch.tensor(tensor_data, dtype=torch.float64, requires_grad=diff), \
                        torch.tensor(tensor_data, dtype=torch.float64, requires_grad=diff)
                else:
                    self.lb, self.ub = \
                        torch.tensor(lb, dtype=torch.float64, requires_grad=diff), \
                        torch.tensor(ub, dtype=torch.float64, requires_grad=diff)
                self.splits = list()
            else:
                self.splits = list()
                abst_shape = list()
                for i, shape_i in enumerate(tensor_shape):
                    if stride[i] == -1:
                        abst_shape.append(1)
                        self.splits.append([0])
                    else:
                        abst_shape.append(int((shape_i + stride[i] - 1) / stride[i]))
                        self.splits.append([stride[i] * x for x in range(abst_shape[-1])])

                if from_init and tensor_data is not None:
                    try:
                        tensor_data = tensor_data.reshape(tensor_shape)
                    except Exception:
                        raise Exception(
                            f'Variable {var_name}: tensor data (shape:{tensor_data.shape}) cannot be casted to required shape({tensor_shape})')

                    lb_data, ub_data = self.summarize_data_and_assign(tensor_data, self.splits)
                    lb_data = np.array(lb_data, dtype=np.float64)
                    ub_data = np.array(ub_data, dtype=np.float64)
                    if from_init_margin > 1e-6:
                        lb_min = np.min(lb_data)
                        ub_max = np.max(ub_data)
                        delta = (ub_max - lb_min) * from_init_margin
                        lb_data -= delta
                        ub_data += delta
                    self.lb, self.ub = \
                        torch.tensor(lb_data, dtype=torch.float64, requires_grad=diff), \
                        torch.tensor(ub_data, dtype=torch.float64, requires_grad=diff)

                else:
                    self.lb, self.ub = \
                        torch.tensor(lb * np.ones(abst_shape), dtype=torch.float64, requires_grad=diff), \
                        torch.tensor(ub * np.ones(abst_shape), dtype=torch.float64, requires_grad=diff)

            self.shape = tensor_shape

        self.var_name = var_name
        if cuda:
            self.lb = self.lb.cuda()
            self.ub = self.ub.cuda()

        return self

    def smash(self, inplace=True):
        """
        smash the abstraction into a single value
        :param inplace:
        :return: None if inplace=True, otherwise, an abstraction of a single value
        """
        # TODO the flow of 1. check scalar, 2. calculate info, 3. update info can be modularized
        if len(self.splits) == 0:
            if inplace:
                return True
            else:
                new_abst = Abstraction()
                new_abst.lb = self.lb
                new_abst.ub = self.ub
                new_abst.splits = self.splits
                new_abst.shape = self.shape
                new_abst.var_name = self.var_name + '_smash'
                return new_abst

        lb = torch.amin(self.lb)
        ub = torch.amax(self.ub)
        new_splits = [[0]] * len(self.shape)
        new_lb, new_ub = torch.ones([1] * len(self.shape)) * lb, torch.ones([1] * len(self.shape)) * ub

        if inplace:
            self.lb, self.ub = new_lb, new_ub
            self.splits = new_splits
        else:
            new_abst = Abstraction()
            new_abst.lb = new_lb
            new_abst.ub = new_ub
            new_abst.splits = new_splits
            new_abst.shape = self.shape
            new_abst.var_name = self.var_name + '_smash'
            return new_abst

    def split_by(self, ref_splits, inplace=True):
        """
            Further split the Abstraction tensor by reference splitting points
        :param ref_splits:
        :param inplace:
        :return:
        """
        assert len(ref_splits) == len(self.splits)
        if len(self.splits) == 0:
            # nothing to do
            if inplace:
                return
            else:
                new_abst = Abstraction()
                new_abst.lb = self.lb
                new_abst.ub = self.ub
                new_abst.splits = self.splits
                new_abst.shape = self.shape
                new_abst.var_name = self.var_name + '_split'
                return new_abst

        new_lb, new_ub = self.lb, self.ub
        new_splits = list()
        for i, old_s in enumerate(self.splits):
            ref_s = ref_splits[i]

            if len(ref_s) < self.shape[i]:

                if len(ref_s) == len(self.splits[i]) and all([x == y for x, y in zip(ref_s, self.splits[i])]):
                    new_splits.append(self.splits[i])
                else:

                    p1 = p2 = 0
                    len1 = len(old_s)
                    len2 = len(ref_s)
                    new_s = list()
                    new_index = list()
                    while p1 < len1 or p2 < len2:
                        if p1 < len1 and (p2 == len2 or old_s[p1] <= ref_s[p2]):
                            if len(new_s) == 0 or old_s[p1] > new_s[-1]:
                                new_s.append(old_s[p1])
                                new_index.append(p1)
                            p1 += 1
                        else:
                            if (len(new_s) == 0 or ref_s[p2] > new_s[-1]) and (ref_s[p2] < self.shape[i]):
                                new_s.append(ref_s[p2])
                                if (p1 < len1) and (ref_s[p2] >= old_s[p1]):
                                    new_index.append(p1)
                                else:
                                    new_index.append(p1 - 1)
                            p2 += 1
                    # print(new_s)
                    # print(new_index)

                    new_lb = torch.index_select(new_lb, i, torch.tensor(new_index).to(new_lb.device))
                    new_ub = torch.index_select(new_ub, i, torch.tensor(new_index).to(new_ub.device))
                    new_splits.append(new_s)

            else:
                # split to atomic level, a simple acceleration
                tmp_s = old_s + [self.shape[i]]
                new_index = [i for i, item in enumerate(old_s) for j in range(tmp_s[i + 1] - tmp_s[i])]

                if len(new_index) > 0:
                    new_lb = torch.index_select(new_lb, i, torch.tensor(new_index).to(new_lb.device))
                    new_ub = torch.index_select(new_ub, i, torch.tensor(new_index).to(new_ub.device))
                    new_splits.append(list(range(self.shape[i])))
                else:
                    new_splits.append(list())

        if inplace:
            self.lb, self.ub = new_lb, new_ub
            self.splits = new_splits
        else:
            new_abst = Abstraction()
            new_abst.lb = new_lb
            new_abst.ub = new_ub
            new_abst.splits = new_splits
            new_abst.shape = self.shape
            new_abst.var_name = self.var_name + '_split'
            return new_abst

    def extend_dim(self, targ_dim, inplace=True):
        """
            Add several singleton dims to abstraction tensors
            Inplace or not
        :param targ_dim:
        :param inplace:
        :return:
        """

        if inplace:
            targ = self
        else:
            targ = Abstraction()
            targ.lb = self.lb
            targ.ub = self.ub
            targ.splits = self.splits
            targ.shape = self.shape
            targ.var_name = self.var_name

        # print(targ.get_dim(), targ_dim)

        while targ_dim - targ.get_dim() > 0:
            targ.lb = targ.lb.unsqueeze(dim=0)
            targ.ub = targ.ub.unsqueeze(dim=0)
            targ.shape = [1] + targ.shape
            targ.splits = [[0]] + targ.splits
        targ.var_name = targ.var_name + '_extend_dim'
        return targ

    def force_resplit(self, new_splits, inplace=True):
        """
            Forced resplitting of current Abstraction tensor with new_splits
            It might be costly and make the abstraction loose because we need to find out all the covered old tensors,
                and query the minimum and maximum to construct each cell of the new Abstraction tensor
        :param new_splits:
        :param inplace:
        :return:
        """

        now_lb, now_ub = self.lb, self.ub

        for dim, (old_split, new_split) in enumerate(zip(self.splits, new_splits)):
            # print(dim, len(old_split), len(new_split))

            if len(old_split) == len(new_split) and all([x == y for x, y in zip(old_split, new_split)]):
                # skip this dim
                continue

            ts_list_lb, ts_list_ub = list(), list()
            for i, l in enumerate(new_split):
                if l == new_split[-1]:
                    r = self.shape[dim]
                else:
                    r = new_split[i + 1]
                old_l = bisect.bisect_right(old_split, l) - 1
                old_r = bisect.bisect_left(old_split, r)
                ts_list_lb.append(
                    now_lb.index_select(dim=dim, index=torch.tensor(range(old_l, old_r)).to(now_lb.device)).amin(
                        dim=dim))
                ts_list_ub.append(
                    now_ub.index_select(dim=dim, index=torch.tensor(range(old_l, old_r)).to(now_ub.device)).amax(
                        dim=dim))

            now_lb = torch.stack(ts_list_lb, dim=dim)
            now_ub = torch.stack(ts_list_ub, dim=dim)

        if inplace:
            self.lb, self.ub = now_lb, now_ub
            self.splits = new_splits.copy()
            self.var_name = self.var_name + '_force_resplit'
            ans = self
        else:
            ans = Abstraction()
            ans.lb, ans.ub = now_lb, now_ub
            ans.splits = new_splits.copy()
            ans.var_name = self.var_name + '_force_resplit'
            ans.shape = self.shape

        return ans

    def is_exact(self, eps=1e-5):
        """
            Return whether the abstraction is precise
        :param eps:
        :return:
        """
        local_lb = self.lb.cpu().detach().reshape(-1).float()
        local_ub = self.ub.cpu().detach().reshape(-1).float()
        dif = torch.norm(local_ub - local_lb).item()
        return dif < eps

    def is_atomic(self):
        return all([len(x) == y for x, y in zip(self.splits, self.shape)])

    def is_empty(self):
        return any([s == 0 for s in self.shape])

    def get_tensor_numel(self):
        return get_numel(self.shape)

    def get_abst_tensor_numel(self):
        # return get_numel([len(x) for x in self.splits])
        return torch.numel(self.lb)

    @staticmethod
    def summarize_data_and_assign(tensor_data, splits, dim=0):
        """
            this method aggregate the min and max of tensor_data according to "splits" blocking rule,
            then construct abstraction from these mins and maxs
        """
        # print(f'split and assign {tensor_data} {splits} dim={dim}')
        if dim == len(splits):
            return np.min(tensor_data), np.max(tensor_data)
        else:
            s_tensor_data = np.split(tensor_data, splits[dim][1:], axis=dim)
            # print(s_tensor_data)
            res = [Abstraction.summarize_data_and_assign(block, splits, dim + 1) for block in s_tensor_data]
            return [item[0] for item in res], [item[1] for item in res]

    def print(self):
        print('===ABST PRINT BEGIN===')
        print('var_name:', self.var_name)
        print('lb_tensor:', self.lb)
        print('lb_tensor shape:', self.lb.shape if not isinstance(self.lb, list) else [item.shape for item in self.lb])
        print('ub_tensor:', self.ub)
        print('ub_tensor shape:', self.ub.shape if not isinstance(self.ub, list) else [item.shape for item in self.ub])
        print('splits:', self.splits)
        print('shape:', self.shape)
        print('===ABST PRINT END===')

    def get_dim(self):
        return len(self.shape)


class Interpreter(object):
    """
        The general class for generating interpretations
    """

    def __init__(self, smash_thres=-1, ceil='precise', floor='precise',
                 loop_constant_abst_cfg=AbstractionInitConfig(diff=False, from_init=True, stride=1),
                 average_pool_mode='precise'):
        # default smash threshold
        self.smash = smash_thres

        # whether to propagate precise or coarse bound for ceil or floor
        # the precise means that, ceil/floor exactly applies ceil/floor func, but this way we cannot have the gradient
        # the idential means that, we just propagate the identical value as the approximation and we can obtain the gradient
        # the coarse means that, the corase bound is (lb, ub+1) for ceil and (lb-1, ub) for floor, which is imprecise but we can obtain the gradient
        assert ceil in ['precise', 'identical', 'coarse']
        assert floor in ['precise', 'identical', 'coarse']
        assert average_pool_mode in ['precise', 'coarse']
        self.ceil = ceil
        self.floor = floor
        self.loop_constant_abst_cfg = loop_constant_abst_cfg
        self.average_pool_mode = average_pool_mode

    def handle(self, abstracts, node, optype, var_name):
        """
            The dispatcher
        :param abstracts:
        :param node:
        :param var_name:
        :return: Abstraction instance, and list of exceptions
        """

        try:
            # print(optype)
            func = getattr(self, 'interp_' + optype)
        except Exception as e:
            print(f'unsupported interpretation for optype {optype}')
            return None, list()

        return func(abstracts, node, optype, var_name)

    def interp_Sub(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        ans.lb = abst0.lb - abst1.ub
        ans.ub = abst0.ub - abst1.lb
        ans.var_name = var_name

        return ans, list()

    def interp_Add(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        ans.lb = abst0.lb + abst1.lb
        ans.ub = abst0.ub + abst1.ub
        ans.var_name = var_name

        return ans, list()

    def interp_Mul(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        choices = torch.stack([abst0.lb * abst1.lb, abst0.lb * abst1.ub, abst0.ub * abst1.lb, abst0.ub * abst1.ub],
                              dim=0)
        ans.lb = torch.amin(choices, dim=0)
        ans.ub = torch.amax(choices, dim=0)
        ans.var_name = var_name

        return ans, list()

    def interp_Pow(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        # x^y, if x can be 0 and y < 0
        if ((abst0.lb <= PossibleNumericalError.UNDERFLOW_LIMIT) &
            (abst0.ub >= -PossibleNumericalError.UNDERFLOW_LIMIT) &
            (abst1.lb < -PossibleNumericalError.UNDERFLOW_LIMIT)).any():
            return None, [PossibleNumericalError(optype, var_name, [abst0.lb, abst0.ub],
                                                 PossibleNumericalError.ERROR_CONTAINS_ZERO)]
        choices = torch.stack(
            [abst0.lb.pow(abst1.lb), abst0.lb.pow(abst1.ub), abst0.ub.pow(abst1.lb), abst0.ub.pow(abst1.ub)],
            dim=0)
        # use float32 to see if the pow operator triggers a numerical error
        # TODO: adapt to the operator's original type,
        # TODO: E.g., if pow(x, y)'s argument x and y are both float64, then we shouldn't generate an exception
        choices_float32 = torch.stack(
            [abst0.lb.to(torch.float32).pow(abst1.lb.to(torch.float32)),
             abst0.lb.to(torch.float32).pow(abst1.ub.to(torch.float32)),
             abst0.ub.to(torch.float32).pow(abst1.lb.to(torch.float32)),
             abst0.ub.to(torch.float32).pow(abst1.ub.to(torch.float32))],
            dim=0)
        if any(PossibleNumericalError.is_invalid(x) for x in choices_float32):
            return None, [PossibleNumericalError(optype, var_name, [abst1.lb, abst1.ub],
                                                 PossibleNumericalError.ERROR_OVERFLOW)]
        ans.lb = torch.amin(choices, dim=0)
        ans.ub = torch.amax(choices, dim=0)
        ans.var_name = var_name

        return ans, list()

    def interp_Div(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)
        if ((abst1.lb <= PossibleNumericalError.UNDERFLOW_LIMIT) & (
                abst1.ub >= -PossibleNumericalError.UNDERFLOW_LIMIT)).any():
            return None, [
                PossibleNumericalError(optype, var_name, [abst1.lb, abst1.ub],
                                       PossibleNumericalError.ERROR_CONTAINS_ZERO)]

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        choices = torch.stack([abst0.lb / abst1.lb, abst0.lb / abst1.ub, abst0.ub / abst1.lb, abst0.ub / abst1.ub],
                              dim=0)
        ans.lb = torch.amin(choices, dim=0)
        ans.ub = torch.amax(choices, dim=0)
        # if PossibleNumericalError.is_invalid(ans.lb) or PossibleNumericalError.is_invalid(ans.ub):
        #     return None, PossibleNumericalError(optype, var_name, [abst1.lb, abst1.ub],
        #                                         PossibleNumericalError.ERROR_CONTAINS_ZERO)
        ans.var_name = var_name

        return ans, list()

    def interp_MatMul(self, abstracts, node, optype, var_name):
        abstA, abstB = abstracts[0], abstracts[1]
        assert isinstance(abstA, Abstraction)
        assert isinstance(abstB, Abstraction)

        if abstA.get_dim() == 1:
            abstA = abstA.split_by([abstB.splits[-2]], inplace=False)
            coeff = np.array(abstA.splits[0][1:] + [abstA.shape[0]]) - np.array(abstA.splits[0])
            coeff = torch.tensor(coeff).to(abstA.lb.device)
            abstA.lb = abstA.lb * coeff
            abstA.ub = abstA.ub * coeff
        elif abstB.get_dim() == 1:
            abstB = abstB.split_by([abstA.splits[-1]], inplace=False)
            coeff = np.array(abstB.splits[0][1:] + [abstB.shape[0]]) - np.array(abstB.splits[0])
            coeff = torch.tensor(coeff).to(abstB.lb.device)
            abstB.lb = abstB.lb * coeff
            abstB.ub = abstB.ub * coeff
        else:
            target_splits = abstA.splits.copy()
            target_splits[-1] = abstB.splits[-2]
            target_splits[:-2] = abstB.splits[:-2]
            abstA = abstA.split_by(target_splits, inplace=False)

            target_splits = abstB.splits.copy()
            target_splits[-2] = abstA.splits[-1]
            target_splits[:-2] = abstA.splits[:-2]
            abstB = abstB.split_by(target_splits, inplace=False)

            coeff = np.array(abstA.splits[-1][1:] + [abstA.shape[-1]]) - np.array(abstA.splits[-1])
            coeff = torch.tensor(coeff).to(abstA.lb.device)
            abstA.lb = abstA.lb * coeff
            abstA.ub = abstA.ub * coeff

        ans = Abstraction()
        ans.lb = torch.minimum(torch.matmul(abstA.lb, abstB.lb),
                               torch.minimum(torch.matmul(abstA.lb, abstB.ub),
                                             torch.minimum(torch.matmul(abstA.ub, abstB.lb),
                                                           torch.matmul(abstA.ub, abstB.ub))))

        ans.ub = torch.maximum(torch.matmul(abstA.lb, abstB.lb),
                               torch.maximum(torch.matmul(abstA.lb, abstB.ub),
                                             torch.maximum(torch.matmul(abstA.ub, abstB.lb),
                                                           torch.matmul(abstA.ub, abstB.ub))))
        ans.var_name = var_name

        if abstA.get_dim() == 1:
            ans.shape = abstB.shape[:-2] + [abstB.shape[-1]]
            ans.splits = abstB.splits[:-2] + [abstB.splits[-1]]
        elif abstB.get_dim() == 1:
            ans.shape = abstA.shape[:-1]
            ans.splits = abstA.splits[:-1]
        else:
            ans.shape = abstA.shape[:-1] + [abstB.shape[-1]]
            now_splits = list()
            for i in range(abstA.get_dim() - 2):
                if abstA.shape[i] >= abstB.shape[i]:
                    now_splits.append(abstA.splits[i])
                else:
                    now_splits.append(abstB.splits[i])
            now_splits.extend([abstA.splits[-2], abstB.splits[-1]])
            ans.splits = now_splits

        # print('A = ')
        # print(abstracts[0].print())
        # print('B = ')
        # print(abstracts[1].print())
        # print('A @ B = ')
        # print(ans.print())

        return ans, list()

    def interp_Gemm(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        alpha = attr.get('alpha', 1.0)
        beta = attr.get('beta', 1.0)
        transA = attr.get('transA', 0) > 0
        transB = attr.get('transB', 0) > 0

        A = abstracts[0]
        B = abstracts[1]
        if len(abstracts) > 2:
            C = abstracts[2]
        else:
            C = None

        if transA:
            newA = Abstraction()
            newA.lb, newA.ub = A.lb.transpose(0, 1), A.ub.transpose(0, 1)
            newA.shape = A.shape[::-1]
            newA.splits = A.splits[::-1]
            newA.var_name = A.var_name + '_transpose'
            A = newA
        if transB:
            newB = Abstraction()
            newB.lb, newB.ub = B.lb.transpose(0, 1), B.ub.transpose(0, 1)
            newB.shape = B.shape[::-1]
            newB.splits = B.splits[::-1]
            newB.var_name = B.var_name + '_transpose'
            B = newB


        target_splits = A.splits.copy()
        target_splits[1] = B.splits[0]
        A = A.split_by(target_splits, inplace=False)

        target_splits = B.splits.copy()
        target_splits[0] = A.splits[1]
        B = B.split_by(target_splits, inplace=False)

        coeff = np.array(A.splits[1][1:] + [A.shape[-1]]) - np.array(A.splits[1])
        coeff = torch.tensor(coeff).to(A.lb.device)
        A.lb = A.lb * coeff
        A.ub = A.ub * coeff

        ans = Abstraction()
        ans.lb = torch.minimum(torch.matmul(A.lb, B.lb),
                               torch.minimum(torch.matmul(A.lb, B.ub),
                                             torch.minimum(torch.matmul(A.ub, B.lb),
                                                           torch.matmul(A.ub, B.ub))))

        ans.ub = torch.maximum(torch.matmul(A.lb, B.lb),
                               torch.maximum(torch.matmul(A.lb, B.ub),
                                             torch.maximum(torch.matmul(A.ub, B.lb),
                                                           torch.matmul(A.ub, B.ub))))

        if alpha >= 0.:
            ans.lb, ans.ub = ans.lb * alpha, ans.ub * alpha
        else:
            ans.lb, ans.ub = ans.ub * alpha, ans.lb * alpha

        ans.splits = [A.splits[0], B.splits[1]]
        ans.shape = [A.shape[0], B.shape[1]]

        if C is not None:
            C = C.extend_dim(ans.get_dim(), inplace=False)

            if beta >= 0.:
                C.lb, C.ub = C.lb * beta, C.ub * beta
            else:
                C.lb, C.ub = C.ub * beta, C.lb * beta

            ans.split_by(C.splits, inplace=True)
            C.split_by(ans.splits, inplace=True)

            ans_new = Abstraction()
            ans_new.shape, ans_new.splits = get_shape_split_with_broadcasting(ans, C)
            ans_new.lb = ans.lb + C.lb
            ans_new.ub = ans.ub + C.ub

            ans = ans_new

        ans.var_name = var_name

        return ans, list()

    def interp_MaxPool(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        need_index = len(node.output) > 1  # need we return indices? # I returned a very coarse index range
        X = Abstraction()
        X.lb = abstracts[0].lb
        X.ub = abstracts[0].ub
        X.splits = abstracts[0].splits.copy()
        X.shape = abstracts[0].shape.copy()
        X.var_name = 'X'
        dim_for_pool = len(X.shape) - 2
        strides = attr.get('strides', [1] * dim_for_pool)
        dilations = attr.get('dilations', [1] * dim_for_pool)
        ceil_mode = attr.get('ceil_mode', 0) == 1
        kernel_shape = attr.get('kernel_shape', None)
        storage_order = attr.get('storage_order', 0)  # TODO: what's the usage of storage_order?
        assert kernel_shape is not None
        auto_pad = attr.get('auto_pad', None)
        pads = attr.get('pads', None)
        if auto_pad is None and pads is None:
            pads = [0] * (2 * dim_for_pool)

        out_shape, padding = compute_outshape_padding(attr.get('auto_pad', 'NOTSET'), pads,
                                                      X.shape[2:], kernel_shape,
                                                      dilations, strides, ceil_mode=ceil_mode)
        padding = np.array(padding)

        """append the padding to the abstraction, so that all conv becomes valid padding ops"""
        add_padding_to_X(X, padding, float("-Inf"))

        """compute mapping to abst indices after conv"""
        new_X_lb, new_X_ub, lpses, rpses = fold_repeated_indices(X, dim_for_pool, out_shape, kernel_shape, dilations,
                                                                 strides)

        """core pool operation"""
        if dim_for_pool == 1:
            func = torch.nn.functional.max_pool1d
        elif dim_for_pool == 2:
            func = torch.nn.functional.max_pool2d
        elif dim_for_pool == 3:
            func = torch.nn.functional.max_pool3d
        else:
            raise NotImplementedError("No max_poolXd for X > 3!")

        Cmin = func(new_X_lb, kernel_size=kernel_shape, stride=strides, padding=0, dilation=dilations,
                    ceil_mode=ceil_mode == 1)
        Cmax = func(new_X_ub, kernel_size=kernel_shape, stride=strides, padding=0, dilation=dilations,
                    ceil_mode=ceil_mode == 1)

        """infer splits and shape"""
        splits = [list() for _ in range(dim_for_pool)]
        for index in range(dim_for_pool):
            lpxs = lpses[index]
            rpxs = rpses[index]
            for i in range(len(lpxs)):
                if i == 0 or (lpxs[i] != rpxs[i]) or (lpxs[i] != lpxs[i - 1]):
                    splits[index].append(i)

        try:
            for i in range(dim_for_pool):
                assert (Cmin.shape[2 + i] == len(splits[i]))
                assert (Cmax.shape[2 + i] == len(splits[i]))
        except Exception:
            print(f'! shape does not match: expected ({[len(x) for x in splits]})')
            print(f'                        got ({[Cmin.shape[i + 2] for i in range(dim_for_pool)]})')

        ans = Abstraction()
        ans.lb = Cmin
        ans.ub = Cmax
        ans.shape = X.shape[:2] + out_shape
        ans.splits = X.splits[:2] + splits
        ans.var_name = var_name

        if need_index:
            ans.var_name = node.output[0]
            indices_abst = Abstraction()
            indices_abst.lb = torch.zeros((1,) * ans.get_dim(), device=ans.lb.device)
            indices_abst.ub = torch.zeros((1,) * ans.get_dim(), device=ans.lb.device) + max(reduce(lambda x, y: x * y, ans.shape) - 1, 0)
            indices_abst.shape = ans.shape.copy()
            indices_abst.splits = [[0] for _ in ans.shape]
            indices_abst.var_name = node.output[1]

        return ans if not need_index else [ans, indices_abst], list()

    def interp_AveragePool(self, abstracts, node, optype, var_name):
        """
            @TODO: for precise mode, need to handle these cases later
                1. count_include_pad = False, left pad < right pad
                2. count_include_pad = False, left pad > right pad, stride > 1
        """
        attr = parse_attribute(node)
        X = Abstraction()
        X.lb = abstracts[0].lb
        X.ub = abstracts[0].ub
        X.splits = abstracts[0].splits.copy()
        X.shape = abstracts[0].shape.copy()
        X.var_name = 'X'
        dim_for_pool = len(X.shape) - 2
        strides = attr.get('strides', [1] * dim_for_pool)
        dilations = [1] * dim_for_pool
        ceil_mode = attr.get('ceil_mode', 0) == 1
        kernel_shape = attr.get('kernel_shape', None)
        count_include_pad = attr.get('count_include_pad', 0) == 1
        assert kernel_shape is not None
        auto_pad = attr.get('auto_pad', None)
        pads = attr.get('pads', None)
        if auto_pad is None and pads is None:
            pads = [0] * (2 * dim_for_pool)

        out_shape, padding = compute_outshape_padding(attr.get('auto_pad', 'NOTSET'), pads,
                                                      X.shape[2:], kernel_shape,
                                                      dilations, strides, ceil_mode=ceil_mode)
        padding = np.array(padding)

        if self.average_pool_mode == 'precise':
            """append the padding to the abstraction, so that all conv becomes valid padding ops"""
            if count_include_pad:
                add_padding_to_X(X, padding, 0)
            else:
                add_padding_to_X(X, padding, float('NAN'))

            """compute mapping to abst indices after conv"""
            new_X_lb, new_X_ub, lpses, rpses = fold_repeated_indices(X, dim_for_pool, out_shape, kernel_shape, dilations,
                                                                     strides)

            """exclude NAN paddings and interpret them as padding"""
            nan_begin_paddings = []
            for i in range(dim_for_pool):
                nan_begin_paddings.append(0)
                indices = [slice(None)] * len(new_X_lb.shape)
                for j in range(0, new_X_lb.shape[2 + i]):
                    indices[2 + i] = j
                    if new_X_lb[tuple(indices)].isnan().all():
                        nan_begin_paddings[-1] = j + 1
                    else:
                        break

                if nan_begin_paddings[-1] > 0:
                    indices = [slice(None)] * len(new_X_lb.shape)
                    indices[2 + i] = slice(nan_begin_paddings[-1], None)
                    new_X_lb = new_X_lb[tuple(indices)]
                    new_X_ub = new_X_ub[tuple(indices)]

            nan_end_paddings = []
            for i in range(dim_for_pool):
                nan_end_paddings.append(0)
                indices = [slice(None)] * len(new_X_lb.shape)
                for j in range(0, new_X_lb.shape[2 + i]):
                    indices[2 + i] = new_X_lb.shape[2 + i] - j - 1
                    if new_X_lb[tuple(indices)].isnan().all():
                        nan_end_paddings[-1] = j + 1
                    else:
                        break

                if nan_end_paddings[-1] > 0:
                    indices = [slice(None)] * len(new_X_lb.shape)
                    indices[2 + i] = slice(None, -nan_end_paddings[-1])
                    new_X_lb = new_X_lb[tuple(indices)]
                    new_X_ub = new_X_ub[tuple(indices)]

            assert not new_X_lb.isnan().any()
            if not count_include_pad:
                if any(x < y for x, y in zip(nan_begin_paddings, nan_end_paddings)):
                    raise NotImplementedError("odd SAME_UPPER not implemented for count_include_pad == 1")
            nan_paddings = [max(x, y) for x, y in zip(nan_begin_paddings, nan_end_paddings)]

            """core pool operation"""
            if dim_for_pool == 1:
                func = torch.nn.functional.avg_pool1d
            elif dim_for_pool == 2:
                func = torch.nn.functional.avg_pool2d
            elif dim_for_pool == 3:
                func = torch.nn.functional.avg_pool3d
            else:
                raise NotImplementedError("No convXd for X > 3!")

            Cmin = func(new_X_lb, kernel_size=kernel_shape, stride=strides, padding=nan_paddings,
                        count_include_pad=count_include_pad, ceil_mode=ceil_mode)
            Cmax = func(new_X_ub, kernel_size=kernel_shape, stride=strides, padding=nan_paddings,
                        count_include_pad=count_include_pad, ceil_mode=ceil_mode)

            """fix Cmin Cmax for odd SAME_LOWER"""
            indices = [slice(None)] * 2 + [
                slice(None, nan_end_paddings[i] - nan_begin_paddings[i])
                if nan_begin_paddings[i] - nan_end_paddings[i] >= strides[
                    i]  # only if strides[i] == 1 and nan_begin_paddings[i] > nan_end_paddings[i]
                else slice(None)
                for i in range(dim_for_pool)]
            Cmin = Cmin[tuple(indices)]
            Cmax = Cmax[tuple(indices)]

        elif self.average_pool_mode == 'coarse':

            """append the padding to the abstraction, so that all conv becomes valid padding ops"""
            if count_include_pad:
                add_padding_to_X(X, padding, 0)
            else:
                add_padding_to_X(X, padding, float('NAN'), mode='minmax')

            """compute mapping to abst indices after conv"""
            new_X_lb, new_X_ub, lpses, rpses = fold_repeated_indices(X, dim_for_pool, out_shape, kernel_shape, dilations,
                                                                     strides)

            """core pool operation"""
            if dim_for_pool == 1:
                func = torch.nn.functional.avg_pool1d
            elif dim_for_pool == 2:
                func = torch.nn.functional.avg_pool2d
            elif dim_for_pool == 3:
                func = torch.nn.functional.avg_pool3d
            else:
                raise NotImplementedError("No avg_poolXd for X > 3!")

            Cmin = func(new_X_lb, kernel_size=kernel_shape, stride=strides, padding=0, count_include_pad=False,
                        ceil_mode=ceil_mode == 1)
            Cmax = func(new_X_ub, kernel_size=kernel_shape, stride=strides, padding=0, count_include_pad=False,
                        ceil_mode=ceil_mode == 1)

        else:
            raise NotImplementedError(f'Unknown average pooling abstraction mode: {self.average_pool_mode}')


        """infer splits and shape"""
        splits = [list() for _ in range(dim_for_pool)]
        for index in range(dim_for_pool):
            lpxs = lpses[index]
            rpxs = rpses[index]
            for i in range(len(lpxs)):
                if i == 0 or (lpxs[i] != rpxs[i]) or (lpxs[i] != lpxs[i - 1]):
                    splits[index].append(i)

        try:
            for i in range(dim_for_pool):
                assert (Cmin.shape[2 + i] == len(splits[i]))
                assert (Cmax.shape[2 + i] == len(splits[i]))
        except Exception:
            print(f'! shape does not match: expected ({[len(x) for x in splits]})')
            print(f'                        got ({[Cmin.shape[i + 2] for i in range(dim_for_pool)]})')

        ans = Abstraction()
        ans.lb = Cmin
        ans.ub = Cmax
        ans.shape = X.shape[:2] + out_shape
        ans.splits = X.splits[:2] + splits

        return ans, list()

    def interp_Conv(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)

        X = Abstraction()
        X.lb = abstracts[0].lb
        X.ub = abstracts[0].ub
        X.splits = abstracts[0].splits.copy()
        X.shape = abstracts[0].shape.copy()
        X.var_name = 'X'
        W = abstracts[1]
        W.lb = abstracts[1].lb
        W.ub = abstracts[1].ub
        W.splits = abstracts[1].splits.copy()
        W.shape = abstracts[1].shape.copy()
        W.var_name = 'W'
        B = abstracts[2] if len(abstracts) >= 3 else None

        dim_for_conv = len(X.shape) - 2
        strides = attr.get('strides', [1] * dim_for_conv)
        group = attr.get('group', 1)
        if group == -1 or group == 0: group = 1
        dilations = attr.get('dilations', [1] * dim_for_conv)
        kernel_shape = W.shape[2:]
        out_shape, padding = compute_outshape_padding(attr.get('auto_pad', 'NOTSET'), attr.get('pads', None),
                                                      X.shape[2:], kernel_shape,
                                                      dilations, strides)
        padding = np.array(padding)

        # print(abstracts[0].var_name)
        # print([abst.shape for abst in abstracts])
        # print(attr)
        # print('out_shape:', out_shape)
        # print('padding:', padding)

        """append the padding to the abstraction, so that all conv becomes valid padding ops"""
        add_padding_to_X(X, padding, 0)

        W_ref_channels = list(set([item % (X.shape[1] // group) for item in X.splits[1]]))
        W.split_by([W.splits[0] if B is None else B.splits[0], W_ref_channels] + [list(range(x)) for x in
                                                                                  W.shape[2:]], inplace=True)
        X_ref_channels = [item + now_group * W.shape[1] for now_group in range(group) for item in W.splits[1]]
        X.split_by([X.splits[0], X_ref_channels] + X.splits[2:], inplace=True)
        if B is not None:
            B = B.split_by([W.splits[0]], inplace=False)

        """multiple X with channel strides"""
        channel_strides = (np.array(X.splits[1][1:] + [X.shape[1]]) - np.array(X.splits[1])).reshape(
            *([-1] + [1] * dim_for_conv))
        X.lb = X.lb * torch.tensor(channel_strides, device=X.lb.device)
        X.ub = X.ub * torch.tensor(channel_strides, device=X.ub.device)

        """compute mapping to abst indices after conv"""
        new_X_lb, new_X_ub, lpses, rpses = fold_repeated_indices(X, dim_for_conv, out_shape, kernel_shape, dilations,
                                                                 strides)

        """core conv operation"""
        if dim_for_conv == 1:
            func = torch.nn.functional.conv1d
        elif dim_for_conv == 2:
            func = torch.nn.functional.conv2d
        elif dim_for_conv == 3:
            func = torch.nn.functional.conv3d
        else:
            raise NotImplementedError("No convXd for X > 3!")

        C1 = func(new_X_lb, W.lb, None, stride=strides, padding=0, dilation=dilations,
                  groups=group)
        C2 = func(new_X_lb, W.ub, None, stride=strides, padding=0, dilation=dilations,
                  groups=group)
        C3 = func(new_X_ub, W.lb, None, stride=strides, padding=0, dilation=dilations,
                  groups=group)
        C4 = func(new_X_ub, W.ub, None, stride=strides, padding=0, dilation=dilations,
                  groups=group)
        Cmin = torch.minimum(torch.minimum(torch.minimum(C1, C2), C3), C4)
        Cmax = torch.maximum(torch.maximum(torch.maximum(C1, C2), C3), C4)
        if B is not None:
            Cmin = Cmin + B.lb.reshape(*([-1] + [1] * dim_for_conv))
            Cmax = Cmax + B.ub.reshape(*([-1] + [1] * dim_for_conv))

        # print('out abst shape - min:', Cmin.shape)
        # print('out abst shape - max:', Cmax.shape)

        """infer splits and shape"""
        splits = [list() for _ in range(dim_for_conv)]
        for index in range(dim_for_conv):
            lpxs = lpses[index]
            rpxs = rpses[index]
            for i in range(len(lpxs)):
                if i == 0 or (lpxs[i] != rpxs[i]) or (lpxs[i] != lpxs[i - 1]):
                    splits[index].append(i)

        try:
            for i in range(dim_for_conv):
                assert (Cmin.shape[2 + i] == len(splits[i]))
                assert (Cmax.shape[2 + i] == len(splits[i]))
        except Exception:
            print(f'! shape does not match: expected ({[len(x) for x in splits]})')
            print(f'                        got ({[Cmin.shape[i + 2] for i in range(dim_for_conv)]})')

        ans = Abstraction()
        ans.lb = Cmin
        ans.ub = Cmax
        ans.shape = [X.shape[0], W.shape[0]] + out_shape
        ans.splits = [X.splits[0], W.splits[0]] + splits
        ans.var_name = var_name

        # print('after conv shape:', ans.shape)
        # ans.print()

        return ans, list()

    def interp_ConvTranspose(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)

        X = Abstraction()
        X.lb = abstracts[0].lb
        X.ub = abstracts[0].ub
        X.splits = abstracts[0].splits.copy()
        X.shape = abstracts[0].shape.copy()
        X.var_name = 'X'
        W = abstracts[1]
        W.lb = abstracts[1].lb
        W.ub = abstracts[1].ub
        W.splits = abstracts[1].splits.copy()
        W.shape = abstracts[1].shape.copy()
        W.var_name = 'W'
        B = abstracts[2] if len(abstracts) >= 3 else None

        dim_for_transconv = len(X.shape) - 2
        strides = attr.get('strides', [1] * dim_for_transconv)
        group = attr.get('group', 1)
        if group == -1 or group == 0: group = 1
        dilations = attr.get('dilations', [1] * dim_for_transconv)
        kernel_shape = W.shape[2:]
        out_shape, padding = compute_outshape_padding_trans(attr.get('auto_pad', 'NOTSET'),
                                                            attr.get('pads', None),
                                                            attr.get('output_shape', None),
                                                            attr.get('output_padding', None),
                                                            X.shape[2:], kernel_shape,
                                                            dilations, strides)
        padding = np.array(padding)

        # print(abstracts[0].var_name)
        # print([abst.shape for abst in abstracts])
        # print(attr)
        # print('out_shape:', out_shape)
        # print('padding:', padding)

        """split and align"""
        X_ref_channels = sorted(list(set([item % (X.shape[1] // group) for item in W.splits[0] + X.splits[1]])))
        X_ref_channels = [item + j * X.shape[1] // group for j in range(group) for item in X_ref_channels]
        X.split_by([X.splits[0], X_ref_channels] + X.splits[2:], inplace=True)
        if B is not None:
            W_ref_channels = list(set([item % (W.shape[1]) for item in B.splits[0]]))
        else:
            W_ref_channels = list(range(W.shape[1]))
        W.split_by([X.splits[1], W_ref_channels] + [list(range(x)) for x in W.shape[2:]])
        if B is not None:
            B_ref_channels = [item + now_group * W.shape[1] for now_group in range(group) for item in W.splits[1]]
            B = B.split_by([B_ref_channels], inplace=False)

        """multiple X with channel strides"""
        channel_strides = (np.array(X.splits[1][1:] + [X.shape[1]]) - np.array(X.splits[1])).reshape(
            *([-1] + [1] * dim_for_transconv))
        X.lb = X.lb * torch.tensor(channel_strides, device=X.lb.device)
        X.ub = X.ub * torch.tensor(channel_strides, device=X.ub.device)
        # X.print()

        """compute mapping to abst indices after conv"""
        new_X_lb, new_X_ub, lpses, rpses = fold_repeated_indices_trans(X, dim_for_transconv, kernel_shape, dilations, strides)
        # print(new_X_lb.shape, len(lpses[0]), len(rpses[0]))
        # print(lpses[0])
        # print(rpses[0])

        """core conv operation"""
        if dim_for_transconv == 1:
            func = torch.nn.functional.conv_transpose1d
        elif dim_for_transconv == 2:
            func = torch.nn.functional.conv_transpose2d
        elif dim_for_transconv == 3:
            func = torch.nn.functional.conv_transpose3d
        else:
            raise NotImplementedError("No conv_transposeXd for X > 3!")

        C1 = func(new_X_lb, W.lb, None, stride=strides, padding=0, dilation=dilations,
                  groups=group)
        C2 = func(new_X_lb, W.ub, None, stride=strides, padding=0, dilation=dilations,
                  groups=group)
        C3 = func(new_X_ub, W.lb, None, stride=strides, padding=0, dilation=dilations,
                  groups=group)
        C4 = func(new_X_ub, W.ub, None, stride=strides, padding=0, dilation=dilations,
                  groups=group)
        Cmin = torch.minimum(torch.minimum(torch.minimum(C1, C2), C3), C4)
        Cmax = torch.maximum(torch.maximum(torch.maximum(C1, C2), C3), C4)
        if B is not None:
            Cmin = Cmin + B.lb.reshape(*([-1] + [1] * dim_for_transconv))
            Cmax = Cmax + B.ub.reshape(*([-1] + [1] * dim_for_transconv))

        """infer splits and shape"""
        """Note: current abstraction may be unsound for strides > 1 due to non-continuous index mapping after folding for output"""

        splits = [list() for _ in range(dim_for_transconv)]
        for index in range(dim_for_transconv):
            for i in range(len(lpses[index])):
                now_rx = rpses[index][i] * strides[index] + (kernel_shape[index] - 1) * dilations[index]
                if i == 0:
                    splits[index].extend(list(range(1 + (kernel_shape[index] - 1) * dilations[index])))
                else:
                    splits[index].extend(list(range(now_rx - strides[index] + 1, now_rx + 1)))
        # print(splits)

        try:
            for i in range(dim_for_transconv):
                assert (Cmin.shape[2 + i] == len(splits[i]))
                assert (Cmax.shape[2 + i] == len(splits[i]))
        except Exception:
            print(f'! shape does not match: expected ({[len(x) for x in splits]})')
            print(f'                        got ({[Cmin.shape[i + 2] for i in range(dim_for_transconv)]})')

        ans = Abstraction()
        ans.lb = Cmin
        ans.ub = Cmax
        ans.shape = [X.shape[0], W.shape[1] * group] + [1 + (X.shape[2 + i] - 1) * strides[i] + (kernel_shape[i] - 1) * dilations[i]  for i in range(dim_for_transconv)]
        ans.splits = [X.splits[0], [item + j * W.shape[1] for j in range(group) for item in W.splits[1]]] + splits
        ans.var_name = var_name
        # ans.print()
        # print('padding:', padding)

        """cropping"""
        """append the padding to the abstraction, so that all conv becomes valid padding ops"""
        add_padding_to_X(ans, -padding, 0)

        return ans, list()

    def interp_Pad(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        mode = attr.get('mode', 'constant')
        if not isinstance(mode, str):
            mode = mode.decode('ascii')
        data = abstracts[0]
        ans = Abstraction()
        ans.lb = data.lb
        ans.ub = data.ub
        ans.splits = data.splits
        ans.shape = data.shape.copy()
        ans.var_name = var_name
        pads = abstracts[1]
        if pads.is_exact():
            pads = pads.lb.type(torch.long).detach().cpu().tolist()
        else:
            print('! pads is not exact in Pad, return the original abstraction instead')
            return data, list()
        pads = [pads[i + j * data.get_dim()] for i in range(data.get_dim()) for j in [0, 1]]
        pads = np.array(pads)
        # print(pads)
        if mode == 'constant':
            if len(abstracts) > 2:
                cv = abstracts[2].lb.detach().cpu().item()
                cv_ub = abstracts[2].ub.detach().cpu().item()
            else:
                cv = cv_ub = 0.
            add_padding_to_X(ans, pads, cv, shift=0, value_ub=cv_ub)
        elif mode == 'edge':
            add_padding_to_X(ans, pads, 0., 'edge', shift=0)
        elif mode == 'reflect':
            add_padding_to_X(ans, pads, 0., 'reflect', shift=0)
        else:
            raise Exception(f'! Unknown mode for padding: {mode}')

        return ans, list()

    def interp_Reciprocal(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        if ((abst.lb <= 0) & (abst.ub >= 0)).any():
            return None, [
                PossibleNumericalError(optype, var_name, [abst.lb, abst.ub],
                                       PossibleNumericalError.ERROR_CONTAINS_ZERO)]

        e1, e2 = 1 / abst.lb, 1 / abst.ub
        ans.lb = torch.minimum(e1, e2)
        ans.ub = torch.maximum(e1, e2)
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Sqrt(self, abstracts, node, optype, var_name):
        abst = abstracts[0]

        print(abst.lb)

        ret = None
        if ((abst.lb < 0)).any():
            ret = None
        else:
            ret = Abstraction()
            ret.lb = abst.lb.sqrt()
            ret.ub = abst.ub.sqrt()
            ret.var_name = var_name
            ret.shape = abst.shape.copy()
            ret.splits = abst.splits.copy()

        if (abst.lb <= PossibleNumericalError.UNDERFLOW_LIMIT).any():
            return ret, [
                PossibleNumericalError(optype, var_name, [abst.lb, abst.ub], PossibleNumericalError.ERROR_UNDERFLOW)
            ]
        else:
            return ret, list()

    def interp_Tanh(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        ans.lb = torch.tanh(abst.lb)
        ans.ub = torch.tanh(abst.ub)
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Relu(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        ans.lb = torch.relu(abst.lb)
        ans.ub = torch.relu(abst.ub)
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Softplus(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        ans.lb = torch.nn.functional.softplus(abst.lb)
        ans.ub = torch.nn.functional.softplus(abst.ub)
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Sigmoid(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        ans.lb = torch.sigmoid(abst.lb)
        ans.ub = torch.sigmoid(abst.ub)
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Neg(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        ans.lb = -abst.ub
        ans.ub = -abst.lb
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Exp(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        if (abst.ub >= PossibleNumericalError.OVERFLOW_D * np.log(10)).any():
            return None, [
                PossibleNumericalError(optype, var_name, [abst.lb, abst.ub], PossibleNumericalError.ERROR_OVERFLOW)]
        ans.lb = torch.exp(abst.lb)
        ans.ub = torch.exp(abst.ub)
        # if PossibleNumericalError.is_invalid(ans.lb) or PossibleNumericalError.is_invalid(ans.ub):
        #     return None, [
        #         PossibleNumericalError(optype, var_name, [abst.lb, abst.ub], PossibleNumericalError.ERROR_OVERFLOW)]
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Log(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        if (abst.lb <= PossibleNumericalError.UNDERFLOW_LIMIT).any():
            return None, [
                PossibleNumericalError(optype, var_name, [abst.lb, abst.ub], PossibleNumericalError.ERROR_UNDERFLOW)]
        ans.lb = torch.log(abst.lb)
        ans.ub = torch.log(abst.ub)
        # if PossibleNumericalError.is_invalid(ans.lb) or PossibleNumericalError.is_invalid(ans.ub):
        #     return None, [
        #         PossibleNumericalError(optype, var_name, [abst.lb, abst.ub], PossibleNumericalError.ERROR_UNDERFLOW)]
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Softmax(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        axis = attr.get('axis', -1)
        abst = abstracts[0]
        ans = Abstraction()
        ans.var_name = var_name
        exp_lb = torch.exp(abst.lb - torch.max(abst.ub, dim=axis, keepdim=True)[0])
        exp_ub = torch.exp(abst.ub - torch.max(abst.ub, dim=axis, keepdim=True)[0])
        multiplies = self.cal_multiplies_for_sum(abstracts[0], [axis])
        # inputs: [l1, l2, l3], [u1, u2, u3]
        # softmax_lb = [l1 / (l1 + u2 + u3), ...]
        # softmax_ub = [u1 / (u1 + l2 + l3)]
        ans.lb = exp_lb / (torch.sum(exp_ub * multiplies, dim=axis, keepdim=True) - exp_ub + exp_lb)
        ans.ub = exp_ub / (torch.sum(exp_lb * multiplies, dim=axis, keepdim=True) - exp_lb + exp_ub)

        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_LogSoftmax(self, abstracts, node, optype, var_name):
        softmax, _ = self.interp_Softmax(abstracts, node, 'Softmax', var_name)
        ans, exceps = self.interp_Log([softmax], node, 'LogSoftMax', var_name)
        return ans, exceps

    def interp_Abs(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        # ans.lb = torch.where(abst.lb > 0, abst.lb, torch.zeros_like(abst.lb))
        # a tighter version
        ans.lb = torch.where((abst.lb < 0) * (abst.ub > 0), torch.zeros_like(abst.lb),
                             torch.minimum(abst.lb.abs(), abst.ub.abs()))
        ans.ub = torch.maximum(abst.lb.abs(), abst.ub.abs())
        ans.var_name = var_name
        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Ceil(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        ans.var_name = var_name
        ans.shape = abst.shape
        ans.splits = abst.splits
        if self.ceil == 'precise':
            ans.lb = abst.lb.ceil()
            ans.ub = abst.ub.ceil()
        elif self.ceil == 'identical':
            ans.lb, ans.ub = abst.lb, abst.ub
        else:
            ans.lb = abst.lb
            ans.ub = abst.ub + 1.
        return ans, list()

    def interp_Floor(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        ans.var_name = var_name
        ans.shape = abst.shape
        ans.splits = abst.splits
        if self.floor == 'precise':
            ans.lb = abst.lb.floor()
            ans.ub = abst.ub.floor()
        elif self.floor == 'identical':
            ans.lb, ans.ub = abst.lb, abst.ub
        else:
            ans.lb = abst.lb - 1.
            ans.ub = abst.ub
        return ans, list()

    def interp_Reshape(self, abstracts, node, optype, var_name, smash=-1):
        """
            Currently, we don't support allowzero != 0
        :param abstracts:
        :param node:
        :param optype:
        :param var_name:
        :param smash: if mode == 'flatten_stretch' or 'stretch', whether to apply force_split to smash if numel >= smash;
            particularly, smash == -1 disable this optimization
        :return:
        """
        abst_data = abstracts[0]
        abst_shape = abstracts[1]
        assert isinstance(abst_data, Abstraction)

        assert abst_shape.is_exact() and abst_shape.is_atomic()
        desired_shape = abst_shape.lb.detach().cpu().type(torch.int).tolist()

        # smash policy init
        if smash == -1: smash = self.smash  # inherent the policy from class setting

        # extract the desired shape from the abstraction
        numel = 1
        for item in abst_data.shape: numel *= item
        tmp = numel
        for ind, item in enumerate(desired_shape):
            if item != -1:
                if item == 0:
                    item = abst_data.shape[ind]
                tmp /= item
        desired_shape = [int(tmp) if x == -1 else (abst_data.shape[ind] if x == 0 else x)
                         for ind, x in enumerate(desired_shape)]

        # print('prev    shape:', abst_data.shape)
        # print('desired shape:', desired_shape)

        assert get_numel(abst_data.shape) == get_numel(desired_shape)

        """
            There are three possible cases regarding reshape that need to be processed:
                - flatten: starting from some dimension i and ends until the last dimension, all flatten to one dimension
                - stretch: stretch the last dimension to multiple dimensions
                - flatten_stretch: if not belong to the above two cases, then use the flatten_stretch mode
                    in this case, we may need to apply some heuristics to reduce the granularity via force_split
        """
        mode = 'flatten_stretch'
        start_dim = None
        if abst_data.get_dim() > len(desired_shape) and all(
                [x == y for x, y in zip(abst_data.shape, desired_shape[:-1])]):
            mode = 'flatten'
            start_dim = len(desired_shape) - 1
        elif abst_data.get_dim() < len(desired_shape) and all(
                [x == y for x, y in zip(abst_data.shape[:-1], desired_shape)]):
            mode = 'stretch'
            start_dim = abst_data.get_dim() - 1
        elif abst_data.get_dim() == len(desired_shape) and all(
                [x == y for x, y in zip(abst_data.shape, desired_shape)]):
            # equal shape
            return abst_data, list()
        else:
            mode = 'flatten_stretch'
            start_dim = [x == y for x, y in zip(abst_data.shape, desired_shape)].index(False)

        ans = abst_data
        if mode in ['flatten', 'flatten_stretch']:
            ans = self.general_flatten(ans, start_dim)
        if mode in ['stretch', 'flatten_stretch']:
            ans = self.general_stretch(ans, start_dim, desired_shape)

        # print('prev abst numel:', abst_data.get_abst_tensor_numel())
        # print('now  abst numel:', ans.get_abst_tensor_numel())

        if mode in ['stretch', 'flatten_stretch'] \
                and smash != -1 and ans.get_abst_tensor_numel() >= smash and ans.get_abst_tensor_numel() >= 8. * abst_data.get_abst_tensor_numel():
            """
                smash triggering policy: answer's numel >= threshold, and current operation enlarges the numel by 8 times
                force split policy: choose the dimension that splits the most to shrink by 2, until the numel is within 4 times of input tensor's
            """

            # print('force smashing triggered')

            target_splits = ans.splits.copy()
            while get_numel([len(x) for x in target_splits]) >= 4. * abst_data.get_abst_tensor_numel():
                max_dim = -1
                for dim_i, split in enumerate(target_splits):
                    if max_dim == -1 or len(split) > len(target_splits[max_dim]):
                        max_dim = dim_i
                target_splits[max_dim] = [x for i, x in enumerate(target_splits[max_dim]) if i % 2 == 0]

            ans = ans.force_resplit(target_splits)

        if var_name is not None:
            ans.var_name = var_name

        return ans, list()

    def interp_Flatten(self, abstracts, node, optype, var_name):
        data = abstracts[0]
        attrs = parse_attribute(node)
        axis = attrs.get('axis', 1)
        if axis < 0:
            axis = data.get_dim() + axis
        ans = self.general_flatten(data, axis)
        if var_name is not None:
            ans.var_name = var_name
        return ans, list()

    def interp_Transpose(self, abstracts, node, optype, var_name):
        attrs = parse_attribute(node)
        inp = abstracts[0]
        ans = Abstraction()
        if 'perm' in attrs:
            perm = attrs['perm']
            ans.lb = inp.lb.permute(*perm)
            ans.ub = inp.ub.permute(*perm)
            ans.splits = [inp.splits[x] for x in perm]
            ans.shape = [inp.shape[x] for x in perm]
        else:
            ans.lb = inp.lb.T
            ans.ub = inp.ub.T
            ans.splits = inp.splits[::-1]
            ans.shape = inp.shape[::-1]
        ans.var_name = var_name
        return ans, list()

    def interp_Shape(self, abstracts, node, optype, var_name):
        start = 0
        end = None
        if len(node.attribute) > 0:
            attr = parse_attribute(node)
            start = attr.get('start', 0)
            end = attr.get('end', None)

        in_tensor = abstracts[0]
        in_tensor_shape = in_tensor.shape

        if end is None:
            in_tensor_shape = in_tensor_shape[start:]
        else:
            in_tensor_shape = in_tensor_shape[start: end]

        ans = Abstraction()
        ans.lb = torch.tensor(in_tensor_shape, dtype=torch.float, device=in_tensor.lb.device)
        ans.ub = torch.tensor(in_tensor_shape, dtype=torch.float, device=in_tensor.ub.device)
        ans.shape = [len(in_tensor_shape)]
        ans.splits = [list(range(len(in_tensor_shape)))]
        ans.var_name = var_name

        return ans, list()

    def interp_Cast(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        to_type = datatype_mapping[attr['to']]
        # print(to_type)

        abst = abstracts[0]

        if to_type in unsupported_types:
            # if the type is not supported, rewrite with null abstraction
            ret = create_empty_tensor(abst.lb.device)
            if var_name is not None:
                ret.var_name = var_name
            else:
                ret.var_name = 'null'
        else:
            ret = abst

        return ret, list()

    def interp_Slice(self, abstracts, node, optype, var_name):
        """
            For version >= 10, the axes, starts, ends, steps are inputs
            Previously, they are attributes
        :param abstracts:
        :param node:
        :param optype:
        :param var_name:
        :return:
        """
        attr = parse_attribute(node)
        if 'starts' in attr:
            # version < 10
            starts = attr.get('starts', [])
            ends = attr.get('ends', [])
            axes = attr.get('axes', list(range(len(starts))))
            steps = attr.get('steps', [1 for _ in axes])
        else:
            # version >= 10
            starts = abstracts[1]
            ends = abstracts[2]
            assert starts.is_exact()
            starts = torch.clip(starts.lb.detach().cpu(), min=-index_clip_thres, max=index_clip_thres).type(
                torch.int64).tolist()
            assert ends.is_exact()
            ends = torch.clip(ends.lb.detach().cpu(), min=-index_clip_thres, max=index_clip_thres).type(
                torch.int64).tolist()
            if len(abstracts) >= 4:
                axes = abstracts[3]
                assert axes.is_exact()
                axes = torch.clip(axes.lb.detach().cpu(), min=-index_clip_thres, max=index_clip_thres).type(
                    torch.int64).tolist()
            else:
                axes = list(range(len(starts)))
            if len(abstracts) >= 5:
                steps = abstracts[4]
                assert steps.is_exact()
                steps = torch.clip(steps.lb.detach().cpu(), min=-index_clip_thres, max=index_clip_thres).type(
                    torch.int64).tolist()
            else:
                steps = [1 for _ in axes]

        # print('axes:  ', axes)
        # print('starts:', starts)
        # print('ends:  ', ends)
        # print('steps: ', steps)

        def squeeze_axis(axis: int, refer_abst: Abstraction):
            """
                Construct an abstraction that squeezes the corresponding axis to 0
            :param axis: the axis to squeeze
            :param refer_abst: the reference abstraction where we learn the information for other axes
            :return: ret: Abstraction
            """
            ret = Abstraction()
            ret.shape = refer_abst.shape.copy()
            ret.shape[axis] = 0
            ret.lb = torch.tensor([]).to(refer_abst.lb.device).reshape(ret.shape)
            ret.ub = torch.tensor([]).to(refer_abst.ub.device).reshape(ret.shape)
            ret.splits = refer_abst.splits.copy()
            ret.splits[axis] = list()
            return ret

        now_abst = Abstraction()
        now_abst.shape = abstracts[0].shape.copy()
        now_abst.splits = abstracts[0].splits.copy()
        now_abst.lb = abstracts[0].lb
        now_abst.ub = abstracts[0].ub
        for axis_ind, now_axis in enumerate(axes):
            now_axis = np.clip(now_axis, a_min=-now_abst.get_dim(), a_max=now_abst.get_dim() - 1)
            if now_axis < 0:
                now_axis = now_abst.get_dim() + now_axis
            # now we assure now_axis >= 0

            now_start, now_end, now_step = starts[axis_ind], ends[axis_ind], steps[axis_ind]
            now_start = np.clip(now_start, a_min=-now_abst.shape[now_axis], a_max=now_abst.shape[now_axis] - 1)
            now_end = np.clip(now_end, a_min=-now_abst.shape[now_axis] - 1, a_max=now_abst.shape[now_axis])
            if now_start < 0:
                now_start = now_abst.shape[now_axis] + now_start
            if now_end < 0:
                now_end = now_abst.shape[now_axis] + now_end
                # maybe now_end could be -1 which corresponds to INT_MIN
            # now we assure now_start >= 0, now_end >= -1

            # print('Slice info:', now_axis, now_start, now_end)

            assert now_step != 0
            if now_step == 1:
                now_end = max(now_end, 0)
                if now_start >= now_end:
                    now_abst = squeeze_axis(now_axis, now_abst)
                else:
                    abst_start = bisect.bisect_right(now_abst.splits[now_axis], now_start) - 1
                    abst_end = bisect.bisect_right(now_abst.splits[now_axis], now_end - 1)
                    # [abst_start, abst_end)
                    now_abst.lb = torch.index_select(now_abst.lb, dim=now_axis,
                                                     index=torch.tensor(range(abst_start, abst_end)).to(
                                                         now_abst.lb.device))
                    now_abst.ub = torch.index_select(now_abst.ub, dim=now_axis,
                                                     index=torch.tensor(range(abst_start, abst_end)).to(
                                                         now_abst.ub.device))
                    now_abst.splits[now_axis] = [max(x - now_start, 0) for x in
                                                 now_abst.splits[now_axis][abst_start:abst_end]]
                    now_abst.shape[now_axis] = now_end - now_start
            else:
                selected = list()
                new_splits = list()
                now_abst_index = None
                shape_counter = 0
                for now_real_index in range(now_start, now_end, now_step):
                    update = False
                    if now_abst_index is None:
                        now_abst_index = bisect.bisect_right(now_abst.splits[now_axis], now_real_index) - 1
                        update = True
                    else:
                        if now_step > 1:
                            while now_abst_index < len(now_abst.splits[now_axis]) - 1 and now_abst.splits[now_axis][
                                now_abst_index + 1] <= now_real_index:
                                now_abst_index += 1
                                update = True
                        else:
                            # now_step < 0
                            while now_abst.splits[now_axis][now_abst_index] > now_real_index:
                                now_abst_index -= 1
                                update = True
                    # print(now_real_index, now_abst_index)
                    if update:
                        selected.append(now_abst_index)
                        new_splits.append(shape_counter)
                    shape_counter += 1
                if len(selected) == 0:
                    now_abst = squeeze_axis(now_axis, now_abst)
                else:
                    now_abst.lb = torch.index_select(now_abst.lb, dim=now_axis,
                                                     index=torch.tensor(selected).to(now_abst.lb.device))
                    now_abst.ub = torch.index_select(now_abst.ub, dim=now_axis,
                                                     index=torch.tensor(selected).to(now_abst.ub.device))
                    now_abst.splits[now_axis] = new_splits
                    now_abst.shape[now_axis] = shape_counter

        now_abst.var_name = var_name
        # print('Shape before slice:', abstracts[0].shape)
        # print('Shape after  slice:', now_abst.shape)
        return now_abst, list()

    def interp_Gather(self, abstracts, node, optype, var_name):
        data, indices = abstracts[0], abstracts[1]
        # data.print()
        # indices.print()

        ind_lb = indices.lb.detach().cpu().numpy().astype(np.int)
        ind_ub = indices.ub.detach().cpu().numpy().astype(np.int)
        axis = parse_attribute(node).get('axis', 0)
        axis_split = data.splits[axis]
        cord_lb = np.apply_along_axis(lambda x: bisect.bisect_right(axis_split, x[0]), axis=1,
                                      arr=ind_lb.reshape(-1, 1)) - 1
        cord_ub = np.apply_along_axis(lambda x: bisect.bisect_right(axis_split, x[0]), axis=1,
                                      arr=ind_ub.reshape(-1, 1)) - 1

        if (cord_lb == cord_ub).all():
            # indices = indices.lb.detach().cpu().numpy().astype(np.int)
            # axis = parse_attribute(node).get('axis', 0)
            # axis_split = data.splits[axis]
            # cord = np.apply_along_axis(lambda x: bisect.bisect_right(axis_split, x[0]), axis=1,
            #                            arr=indices.reshape(-1, 1)) - 1
            cord = cord_lb
            cord = cord.reshape(indices.shape)

            # print(axis, indices, indices.shape)
            # print(axis_split)
            if cord.ndim == 0:
                ans = Abstraction()
                slicer = [slice(None, None, None) for _ in range(data.get_dim())]
                slicer[axis] = cord
                ans.lb = data.lb[slicer]
                ans.ub = data.ub[slicer]
                ans.shape = data.shape[:axis] + data.shape[axis + 1:]
                ans.splits = data.splits[:axis] + data.splits[axis + 1:]
                ans.var_name = var_name

            else:

                new_splits = list()

                for i in range(np.array(cord).ndim):
                    now_splits = [(0,)] + [(j + 1,) for j in range(cord.shape[i] - 1) if np.sum(
                        cord[(np.s_[:],) * i + (j,) + (np.s_[:],) * (cord.ndim - i - 1)] !=
                        cord[(np.s_[:],) * i + (j + 1,) + (np.s_[:],) * (cord.ndim - i - 1)]) > 0]
                    new_splits.append(now_splits)

                new_indices_shape = [len(x) for x in new_splits]
                new_indices = reduce(lambda lst, arr: [x + y for x in lst for y in arr], new_splits)
                # new_indices = cord[new_indices].reshape(new_indices_shape)
                # print(new_indices)
                new_indices = np.array([cord[item] for item in new_indices]).reshape(-1)
                # print(new_indices)
                new_indices = torch.tensor(new_indices, dtype=torch.long).to(data.lb.device)

                ans = Abstraction()
                ans.shape = data.shape[:axis] + list(indices.shape) + data.shape[axis + 1:]
                ans.splits = data.splits[:axis] + [[x[0] for x in lst] for lst in new_splits] + data.splits[axis + 1:]
                ans.lb = data.lb.index_select(dim=axis, index=new_indices).reshape([len(item) for item in ans.splits])
                ans.ub = data.ub.index_select(dim=axis, index=new_indices).reshape([len(item) for item in ans.splits])
                ans.var_name = var_name

            # ans.print()

            return ans, list()

        else:

            axis = parse_attribute(node).get('axis', 0)
            lb = data.lb.amin(dim=axis)
            ub = data.ub.amax(dim=axis)

            for _ in range(indices.get_dim()):
                lb = lb.unsqueeze(dim=axis)
                ub = ub.unsqueeze(dim=axis)

            new_shape = data.splits[:axis] + list(indices.shape) + data.shape[axis + 1:]
            new_splits = data.splits[:axis] + [[0] for _ in indices.shape] + data.splits[axis + 1:]

            ans = Abstraction()
            ans.shape = new_shape
            ans.splits = new_splits
            ans.lb = lb
            ans.ub = ub
            ans.var_name = var_name

            return ans, list()

    def interp_GatherND(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        b = attr.get('batch_dims', 0)
        a_data, a_indices = abstracts[0], abstracts[1]
        r = a_data.get_dim()
        q = a_indices.get_dim()

        # a_data.print()
        # a_indices.print()

        # split the block to align
        ref_splits = a_data.splits.copy()
        ref_splits[:b] = a_indices.splits[:b]
        a_data = a_data.split_by(ref_splits, inplace=False)
        ref_splits = a_indices.splits.copy()
        ref_splits[:b] = a_data.splits[:b]
        ref_splits[-1] = list(range(a_indices.shape[-1]))
        a_indices = a_indices.split_by(ref_splits, inplace=False)

        # map the actual indexing to abstract indexing
        ind_lb, ind_ub = a_indices.lb.type(torch.long), a_indices.ub.type(torch.long)
        map_ind = list()
        for k in range(a_indices.shape[-1]):
            nowdim = b + k
            ind_mapping = torch.tensor([bisect.bisect_right(a_data.splits[nowdim], ind) - 1 for ind in range(a_data.shape[nowdim])],
                                       dtype=torch.long, device=ind_lb.device)
            lbind = ind_mapping[ind_lb.select(-1, k).clamp(min=0, max=a_data.shape[nowdim]-1)].unsqueeze(-1)
            ubind = ind_mapping[ind_ub.select(-1, k).clamp(min=0, max=a_data.shape[nowdim]-1)].unsqueeze(-1)
            if (lbind == ubind).all():
                map_ind.append(lbind)
            else:
                a_data.lb = a_data.lb.amin(nowdim, keepdim=True)
                a_data.ub = a_data.ub.amax(nowdim, keepdim=True)
                lbind = torch.zeros_like(lbind, dtype=torch.long, device=lbind.device)
                map_ind.append(lbind)
        map_ind = torch.cat(map_ind, dim=-1)
        # print(map_ind)

        # figure out the index in 1-dimensional array
        K = a_indices.shape[-1]
        multipler = torch.ones(K, dtype=torch.long)
        for i in range(1, r-b):
            multipler[:i] *= a_data.lb.shape[b+i]
        multipler = multipler.to(map_ind.device)
        # print('multipler', multipler)
        onedim_ind = torch.matmul(map_ind, multipler).unsqueeze(-1)
        remainder = 1
        for i in range(b+K, r):
            remainder *= a_data.lb.shape[i]
        onedim_ind = onedim_ind.tile((1,) * (onedim_ind.dim() - 1) + (remainder,))
        onedim_ind += torch.tensor(list(range(remainder)), dtype=torch.long, device=onedim_ind.device)
        # print(onedim_ind.shape, onedim_ind)

        multipler = torch.ones(b, dtype=torch.long)
        for i in range(1, r):
            multipler[:i] *= a_data.lb.shape[i]
        for i in range(b):
            now_multipler = torch.tensor(list(range(a_data.lb.shape[i])), dtype=torch.long, device=onedim_ind.device) * multipler[i]
            for j in range(onedim_ind.dim()):
                if j < i:
                    now_multipler.unsqueeze_(dim=0)
                elif j > i:
                    now_multipler.unsqueeze_(dim=-1)
            # print('b =', i, now_multipler, now_multipler.shape)
            onedim_ind += now_multipler

        # main work
        shape_desiree = a_data.lb.shape[:b] + map_ind.shape[b: -1] + a_data.lb.shape[b+K:]
        ans_lb = a_data.lb.reshape(-1).take(onedim_ind.reshape(-1))
        ans_ub = a_data.ub.reshape(-1).take(onedim_ind.reshape(-1))
        ans_lb = ans_lb.reshape(shape_desiree)
        ans_ub = ans_ub.reshape(shape_desiree)

        ans = Abstraction()
        ans.lb = ans_lb
        ans.ub = ans_ub
        ans.splits = a_data.splits[:b] + a_indices.splits[b: -1] + a_data.splits[b+K:]
        ans.shape = a_data.shape[:b] + a_indices.shape[b: -1] + a_data.shape[b+K:]
        ans.var_name = var_name

        return ans, list()

    def interp_Squeeze(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        if 'axes' in attr:
            axes = attr['axes']
        elif len(abstracts) > 1:
            axes = abstracts[1]
            assert axes.is_exact()
            axes = axes.lb.detach().cpu().type(torch.int64).tolist()
        else:
            axes = [i for i, s in enumerate(abstracts[0].shape) if s == 1][::-1]
        # make sure the axes are deleted from back to front so that the dim indices do not shift
        axes = sorted([i if i >= 0 else abstracts[0].get_dim() + i for i in axes], reverse=True)
        now_abst = Abstraction()
        now_abst.shape = abstracts[0].shape.copy()
        now_abst.splits = abstracts[0].splits.copy()
        now_abst.lb = abstracts[0].lb
        now_abst.ub = abstracts[0].ub
        for axis in axes:
            assert now_abst.shape[axis] == 1
            del now_abst.shape[axis]
            del now_abst.splits[axis]
            now_abst.lb = now_abst.lb.squeeze(dim=axis)
            now_abst.ub = now_abst.ub.squeeze(dim=axis)
        now_abst.var_name = var_name
        return now_abst, list()

    def interp_Unsqueeze(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        if 'axes' in attr:
            axes = attr['axes']
        elif len(abstracts) > 1:
            axes = abstracts[1]
            assert axes.is_exact()
            axes = axes.lb.detach().cpu().type(torch.int64).tolist()
        else:
            axes = [i for i, s in enumerate(abstracts[0].shape) if s == 1][::-1]
        # make sure the axes are deleted from back to front so that the dim indices do not shift
        axes = sorted([i if i >= 0 else abstracts[0].get_dim() + i for i in axes], reverse=True)
        now_abst = Abstraction()
        now_abst.shape = abstracts[0].shape.copy()
        now_abst.splits = abstracts[0].splits.copy()
        now_abst.lb = abstracts[0].lb
        now_abst.ub = abstracts[0].ub
        for axis in axes:
            (now_abst.shape).insert(axis, 1)
            (now_abst.splits).insert(axis, [0])
            now_abst.lb = now_abst.lb.unsqueeze(dim=axis)
            now_abst.ub = now_abst.ub.unsqueeze(dim=axis)
        now_abst.var_name = var_name
        return now_abst, list()

    def interp_ScatterElements(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        axis = attr.get('axis', 0)
        reduction = attr.get('reduction', b'none')
        reduction = reduction.decode('ascii')

        data, indices, updates = abstracts[0], abstracts[1], abstracts[2]
        if axis < 0:
            axis += data.get_dim()

        index_map = [0 for _ in range(data.shape[axis])]
        for item in data.splits[axis]:
            index_map[item] += 1
        p = -1
        for i, item in enumerate(index_map):
            p += item
            index_map[i] = p
        index_map = torch.tensor(index_map, dtype=torch.long, device=indices.lb.device)

        ind_lb = torch.clip(indices.lb.long(), min=0, max=data.shape[axis]-1)
        ind_ub = torch.clip(indices.ub.long(), min=0, max=data.shape[axis]-1)
        ind_new_lb = index_map[ind_lb]
        ind_new_ub = index_map[ind_ub]

        new_indices = Abstraction()
        new_indices.lb = ind_new_lb
        new_indices.ub = ind_new_ub
        new_indices.shape = indices.shape
        new_indices.splits = indices.splits

        ans = Abstraction()

        if new_indices.is_exact():
            # print('!!!!! exact')
            new_indices.split_by(updates.splits, inplace=True)
            ref_splits = new_indices.splits.copy()
            ref_splits[axis] = data.splits[axis]
            data = data.split_by(ref_splits, inplace=False)
            ref_splits = data.splits.copy()
            ref_splits[axis] = new_indices.splits[axis]
            new_indices.split_by(ref_splits, inplace=True)
            updates = updates.split_by(new_indices.splits, inplace=False)

            old_lb, old_ub = data.lb, data.ub
            if reduction == 'none':
                new_lb = torch.scatter(old_lb, axis, new_indices.lb, updates.lb)
                new_ub = torch.scatter(old_ub, axis, new_indices.lb, updates.ub)
            elif reduction == 'add':
                new_lb = torch.scatter_add(old_lb, axis, new_indices.lb, updates.lb)
                new_ub = torch.scatter_add(old_ub, axis, new_indices.lb, updates.ub)
            elif reduction == 'mul':
                new_lb = torch.zeros_like(old_lb) + old_lb
                new_ub = torch.zeros_like(old_ub) + old_ub
                new_lbl = new_lb.scatter_(axis, new_indices.lb, updates.lb, reduce='multiply')
                new_lbu = new_lb.scatter_(axis, new_indices.lb, updates.ub, reduce='multiply')
                new_ubl = new_ub.scatter_(axis, new_indices.lb, updates.lb, reduce='multiply')
                new_ubu = new_ub.scatter_(axis, new_indices.lb, updates.ub, reduce='multiply')
                choices = torch.stack([new_lbl, new_lbu, new_ubl, new_ubu], dim=0)
                new_lb = torch.amin(choices, dim=0)
                new_ub = torch.amax(choices, dim=0)
            ans.lb = torch.minimum(old_lb, new_lb)
            ans.ub = torch.maximum(old_ub, new_ub)
        else:
            # This over-approximation causes discontinuities, may need further improvement.
            ref_splits = data.splits.copy()
            ref_splits[axis] = updates.splits[axis]
            updates = updates.split_by(ref_splits, inplace=False)
            ref_splits = updates.splits.copy()
            ref_splits[axis] = data.splits[axis]
            data = data.split_by(ref_splits, inplace=False)

            old_lb, old_ub = data.lb, data.ub
            updates_min = torch.amin(updates.lb, dim=axis, keepdim=True)
            updates_max = torch.amax(updates.ub, dim=axis, keepdim=True)
            if reduction == 'none':
                new_lb = torch.minimum(old_lb, updates_min)
                new_ub = torch.maximum(old_ub, updates_max)
            elif reduction == 'add':
                new_lb = old_lb + updates_min
                new_ub = old_ub + updates_max
            elif reduction == 'mul':
                new_lbl = old_lb * updates_min
                new_lbu = old_lb * updates_max
                new_ubl = old_ub * updates_min
                new_ubu = old_ub * updates_max
                choices = torch.stack([new_lbl, new_lbu, new_ubl, new_ubu], dim=0)
                new_lb = torch.amin(choices, dim=0)
                new_ub = torch.amax(choices, dim=0)
            ans.lb = torch.minimum(old_lb, new_lb)
            ans.ub = torch.maximum(old_ub, new_ub)

        ans.splits = data.splits
        ans.shape = data.shape
        ans.var_name = var_name
        return ans, list()

    def interp_Expand(self, abstracts, node, optype, var_name):
        input, shape = abstracts[0], abstracts[1]
        assert shape.is_exact() and shape.is_atomic()
        shape = shape.lb.long().detach().cpu().numpy()
        old_shape = shape.copy()
        if len(shape) < input.get_dim():
            shape = input.shape[: input.get_dim() - len(shape)] + shape
        ans = input.extend_dim(len(shape), inplace=False)
        for i in range(len(shape)):
            if shape[i] == 1 and ans.shape[i] != 1:
                shape[i] = ans.shape[i]
            elif shape[i] > 1 and ans.shape[i] == 1:
                ans.shape[i] = shape[i]
            else:
                try:
                    assert shape[i] == ans.shape[i]
                except:
                    raise Exception(f'Expand shape not much: {input.shape} and {old_shape}')
        ans.var_name = var_name
        return ans, list()

    def interp_Concat(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        axis = attr['axis']

        # we need to finer split each Abstraction's other dimensions by other Abstraction's splits to align the shapes
        new_splits = list([set() for _ in abstracts[0].shape])
        for abst in abstracts:
            for dim, split in enumerate(abst.splits):
                if dim != axis:
                    new_splits[dim] = new_splits[dim].union(split)
        new_splits = [sorted(list(item)) for item in new_splits]
        for i, abst in enumerate(abstracts):
            new_splits[axis] = abst.splits[axis]
            abstracts[i] = abst.split_by(new_splits, inplace=False)

        # for i,abst in enumerate(abstracts):
        #     print(i, abst.shape, abst.splits)

        # start the real concatenation
        ret = Abstraction()
        ret.lb = torch.cat([item.lb for item in abstracts], dim=axis)
        ret.ub = torch.cat([item.ub for item in abstracts], dim=axis)
        ret.shape = abstracts[0].shape.copy()
        ret.shape[axis] = sum([abst.shape[axis] for abst in abstracts])
        ret.splits = abstracts[0].splits.copy()
        axis_new_split = list()
        cum = 0
        for i, abst in enumerate(abstracts):
            axis_new_split.extend([j + cum for j in abst.splits[axis]])
            cum += abst.shape[axis]
        ret.splits[axis] = axis_new_split
        ret.var_name = var_name

        return ret, list()

    def interp_Split(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        axis = attr.get('axis', 0)

        if len(abstracts) > 1 and abstracts[1].is_exact():
            split = abstracts[1].lb.detach().cpu().type(torch.long).tolist()
            split = [0] + [x for x in split[:-1]]
        else:
            split = [i * (abstracts[0].shape[axis] // len(node.output)) for i in range(len(node.output))]

        data = abstracts[0]
        ref_splits = data.splits.copy()
        ref_splits[axis] = sorted(list(set(split)))
        data = data.split_by(ref_splits, inplace=False)
        # data.print()

        ans = list()

        p = 0
        for j, s_ind in enumerate(split):
            e_ind = split[j + 1] - 1 if j < len(split) - 1 else data.shape[axis] - 1
            if len(data.splits[axis]) > 0:
                pe = bisect.bisect_right(data.splits[axis], e_ind)
            else:
                pe = 0
            now_ans = Abstraction()
            slices = [slice(None, None) for _ in data.shape]
            slices[axis] = slice(p, pe)
            now_ans.lb = data.lb[slices]
            now_ans.ub = data.ub[slices]
            now_ans.splits = data.splits.copy()
            now_ans.splits[axis] = [x - s_ind for x in data.splits[axis][p: pe]]
            now_ans.shape = data.shape.copy()
            if (now_ans.lb.shape[axis]) > 0:
                now_ans.shape[axis] = data.splits[axis][pe] - data.splits[axis][p] if pe < len(data.splits[axis]) else \
                    data.shape[axis] - data.splits[axis][p]
            else:
                now_ans.shape[axis] = 0
            now_ans.var_name = node.output[j]
            ans.append(now_ans)
            p = pe

        return ans, list()

    def interp_Identity(self, abstracts, node, optype, var_name):
        ret = Abstraction()
        ret.lb = abstracts[0].lb
        ret.ub = abstracts[0].ub
        ret.shape = abstracts[0].shape.copy()
        ret.splits = abstracts[0].splits.copy()
        ret.var_name = var_name
        return ret, list()

    def interp_ConstantOfShape(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        value = attr.get('value', 0)
        device = abstracts[0].lb.device

        if not isinstance(value, int) and not isinstance(value, float):
            value = numpy_helper.to_array(value).reshape(-1)[0]

        if abstracts[0].is_exact():
            ans = Abstraction()
            ans.shape = list(abstracts[0].lb.long().numpy())
            if len(ans.shape) == 0:
                ans.splits = []
                ans.shape = []
                ans.lb = torch.tensor(float(value), dtype=torch.float64, device=device)
                ans.ub = torch.tensor(float(value), dtype=torch.float64, device=device)
            # elif (len(ans.shape) == 1) and ans.shape[0] == 0:
            #     # empty tensor
            #     ans.splits = []
            #     ans.shape = []
            #     ans.lb = torch.zeros((0), dtype=torch.float64, device=device)
            #     ans.ub = torch.zeros((0), dtype=torch.float64, device=device)
            else:
                ans.splits = [[0] if ans.shape[i] > 0 else [] for i in range(len(ans.shape))]
                ans.lb = torch.full([1 if item > 0 else 0 for item in ans.shape], float(value), dtype=torch.float64, device=device)
                ans.ub = torch.full([1 if item > 0 else 0 for item in ans.shape], float(value), dtype=torch.float64, device=device)
            ans.var_name = var_name
            return ans, list()
        else:  # unknown shape
            return None, list()

    def interp_RandomUniformLike(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        low = attr.get('low', 0)
        high = attr.get('high', 1)
        device = abstracts[0].lb.device
        ans = Abstraction()
        ans.shape = abstracts[0].shape.copy()
        ans.splits = [[0] for _ in range(len(ans.shape))]
        ans.lb = torch.full([1] * len(ans.shape), low, device=device)
        ans.ub = torch.full([1] * len(ans.shape), high, device=device)
        ans.var_name = var_name
        return ans, list()

    def interp_RandomNormal(self, abstracts, node, optype, var_name, device=None):
        """use [mu - 5 * sigma, mu + 5 * sigma] as the interval"""
        attr = parse_attribute(node)
        mean = attr.get('mean', 0.0)
        scale = attr.get('scale', 1.0)
        shape = attr['shape']

        ans = Abstraction()
        ans.shape = shape.copy()
        ans.splits = [[] if item == 0 else [0] for item in ans.shape]
        abs_shape = [0 if item == 0 else 1 for item in ans.shape]
        ans.lb = torch.full(abs_shape, mean - 5. * scale)
        ans.ub = torch.full(abs_shape, mean + 5. * scale)
        if device is not None:
            ans.lb, ans.ub = ans.lb.to(device), ans.ub.to(device)
        ans.var_name = var_name
        return ans, list()

    def interp_Range(self, abstracts, node, optype, var_name):
        start, limit, delta = abstracts[0], abstracts[1], abstracts[2]
        if start.is_exact() and limit.is_exact() and delta.is_exact():
            lb = torch.arange(start=start.lb.cpu().item(), end=limit.lb.cpu().item(), step=delta.lb.cpu().item(), device=start.lb.device)
            ub = lb.clone()
            ans = Abstraction()
            ans.lb, ans.ub = lb, ub
            ans.splits = [list(range(lb.shape[0]))]
            ans.shape = [lb.shape[0]]
        else:
            max_range = torch.maximum(torch.abs(start.lb - limit.ub), torch.abs(start.ub - limit.lb))
            if delta.lb < 0. < delta.ub:
                return None, [
                    PossibleNumericalError(optype, var_name, [delta.lb, delta.ub],
                                           PossibleNumericalError.ERROR_CONTAINS_ZERO)]
            min_step = torch.minimum(torch.abs(delta.lb), torch.abs(delta.ub))
            max_element = torch.ceil(max_range / min_step).cpu().long().item()
            ans = Abstraction()
            ans.lb = torch.minimum(start.lb, limit.lb).view(-1)
            ans.ub = torch.maximum(start.ub, limit.ub).view(-1)
            ans.splits = [[0]]
            ans.shape = [max_element]
        ans.var_name = var_name
        return ans, list()

    def interp_Constant(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        print('Constant node', var_name, 'found')
        print('should not appear here --- the value should have been initialized during preprocessing')
        return None, list()

    def interp_Less(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        ones = torch.ones_like(abst0.lb, dtype=abst0.lb.dtype, device=abst0.lb.device)
        zeros = torch.zeros_like(abst0.lb, dtype=abst0.ub.dtype, device=abst0.lb.device)
        ans.lb = torch.where(abst0.ub >= abst1.lb, zeros, ones)
        ans.ub = torch.where(abst0.lb < abst1.ub, ones, zeros)
        ans.var_name = var_name
        return ans, list()

    def interp_LessOrEqual(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        ones = torch.ones_like(abst0.lb, dtype=abst0.lb.dtype, device=abst0.lb.device)
        zeros = torch.zeros_like(abst0.lb, dtype=abst0.ub.dtype, device=abst0.lb.device)
        ans.lb = torch.where(abst0.ub > abst1.lb, zeros, ones)
        ans.ub = torch.where(abst0.lb <= abst1.ub, ones, zeros)
        ans.var_name = var_name
        return ans, list()

    def interp_Greater(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        ones = torch.ones_like(abst0.lb, dtype=abst0.lb.dtype, device=abst0.lb.device)
        zeros = torch.zeros_like(abst0.lb, dtype=abst0.ub.dtype, device=abst0.lb.device)
        ans.lb = torch.where(abst0.lb <= abst1.ub, zeros, ones)
        ans.ub = torch.where(abst0.ub > abst1.lb, ones, zeros)
        ans.var_name = var_name
        return ans, list()

    def interp_GreaterOrEqual(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        ones = torch.ones_like(abst0.lb, dtype=abst0.lb.dtype, device=abst0.lb.device)
        zeros = torch.zeros_like(abst0.lb, dtype=abst0.ub.dtype, device=abst0.lb.device)
        ans.lb = torch.where(abst0.lb < abst1.ub, zeros, ones)
        ans.ub = torch.where(abst0.ub >= abst1.lb, ones, zeros)
        ans.var_name = var_name
        return ans, list()

    def interp_Equal(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        ones = torch.ones_like(abst0.lb, dtype=abst0.lb.dtype, device=abst0.lb.device)
        zeros = torch.zeros_like(abst0.lb, dtype=abst0.ub.dtype, device=abst0.lb.device)
        ans.lb = torch.where((abst0.lb < abst1.ub) + (abst0.ub > abst1.lb), zeros, ones)
        ans.ub = torch.where((abst0.ub < abst1.lb) + (abst0.lb > abst1.ub), zeros, ones)
        ans.var_name = var_name
        return ans, list()

    def interp_Not(self, abstracts, node, optype, var_name):
        ans = Abstraction()
        ans.shape, ans.splits = abstracts[0].shape.copy(), abstracts[0].splits.copy()
        ans.lb = 1 - abstracts[0].ub
        ans.ub = 1 - abstracts[0].lb
        ans.var_name = var_name
        return ans, list()

    def interp_Min(self, abstracts, node, optype, var_name):
        ans = Abstraction()
        ans.shape, ans.splits = abstracts[0].shape.copy(), abstracts[0].splits.copy()
        ans.lb = abstracts[0].lb.clone()
        ans.ub = abstracts[0].ub.clone()
        ans.var_name = var_name
        for abst in abstracts[1:]:
            abst0 = ans.extend_dim(abst.get_dim(), inplace=False)
            abst1 = abst.extend_dim(ans.get_dim(), inplace=False)

            abst0.split_by(abst1.splits, inplace=True)
            abst1.split_by(abst0.splits, inplace=True)

            ans = Abstraction()
            ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
            ans.lb = torch.minimum(abst0.lb, abst1.lb)
            ans.ub = torch.minimum(abst0.ub, abst1.ub)
            ans.var_name = var_name

        return ans, list()

    def interp_Max(self, abstracts, node, optype, var_name):
        ans = Abstraction()
        ans.shape, ans.splits = abstracts[0].shape.copy(), abstracts[0].splits.copy()
        ans.lb = abstracts[0].lb.clone()
        ans.ub = abstracts[0].ub.clone()
        ans.var_name = var_name
        for abst in abstracts[1:]:
            abst0 = ans.extend_dim(abst.get_dim(), inplace=False)
            abst1 = abst.extend_dim(ans.get_dim(), inplace=False)

            abst0.split_by(abst1.splits, inplace=True)
            abst1.split_by(abst0.splits, inplace=True)

            ans = Abstraction()
            ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
            ans.lb = torch.maximum(abst0.lb, abst1.lb)
            ans.ub = torch.maximum(abst0.ub, abst1.ub)
            ans.var_name = var_name

        return ans, list()

    def interp_Sum(self, abstracts, node, optype, var_name):
        ans = Abstraction()
        ans.shape, ans.splits = abstracts[0].shape.copy(), abstracts[0].splits.copy()
        ans.lb = abstracts[0].lb.clone()
        ans.ub = abstracts[0].ub.clone()
        ans.var_name = var_name
        for abst in abstracts[1:]:
            abst0 = ans.extend_dim(abst.get_dim(), inplace=False)
            abst1 = abst.extend_dim(ans.get_dim(), inplace=False)

            abst0.split_by(abst1.splits, inplace=True)
            abst1.split_by(abst0.splits, inplace=True)

            ans = Abstraction()
            ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
            ans.lb = abst0.lb + abst1.lb
            ans.ub = abst0.ub + abst1.ub
            ans.var_name = var_name

        return ans, list()

    def interp_ReduceMin(self, abstracts, node, optype, var_name, op=torch.min):
        attr = parse_attribute(node)
        keepdims = attr.get('keepdims', 1)
        axes = attr.get('axes', None)
        if axes is None:
            axes = list(range(abstracts[0].get_dim()))
        else:
            axes = [(axis + abstracts[0].get_dim()) % abstracts[0].get_dim() for axis in axes]
            axes.sort()
        ans = Abstraction()
        ans.lb = abstracts[0].lb
        ans.ub = abstracts[0].ub
        for d in axes[::-1]:  # first keep the dimension
            ans.lb = op(ans.lb, dim=d, keepdim=keepdims == 1)[0]
            ans.ub = op(ans.ub, dim=d, keepdim=keepdims == 1)[0]

        ans.shape = abstracts[0].shape.copy()
        ans.splits = abstracts[0].splits.copy()
        if keepdims == 1:
            for d in axes:
                ans.shape[d] = 1
                ans.splits[d] = [0]
        else:
            for d in axes:
                ans.shape[d] = None
                ans.splits[d] = None

            ans.shape = list(filter(lambda x: x is not None, ans.shape))
            ans.splits = list(filter(lambda x: x is not None, ans.splits))

        ans.var_name = var_name
        return ans, list()

    def interp_ReduceMax(self, abstracts, node, optype, var_name):
        return self.interp_ReduceMin(abstracts, node, optype, var_name, op=torch.max)

    def interp_ReduceSum(self, abstracts, node, optype, var_name, op=torch.sum):
        attr = parse_attribute(node)
        keepdims = attr.get('keepdims', 1)
        axes = attr.get('axes', None)
        if axes is None:
            noop_with_empty_axes = attr.get('noop_with_empty_axes', 0)
            if noop_with_empty_axes != 0 and len(abstracts) <= 1:
                # noop behavior
                return abstracts[0], list()
            else:
                if len(abstracts) > 1:
                    assert abstracts[1].is_exact()
                    axes = abstracts[1].lb.detach().cpu().type(torch.int64).tolist()
                else:
                    axes = [i for i in range(len(abstracts[0].shape))]
        assert axes is not None
        ans = Abstraction()
        # compute multiplies to calculate reduced sum
        multiplies = self.cal_multiplies_for_sum(abstracts[0], axes)

        ans.lb = op(abstracts[0].lb * multiplies, dim=axes, keepdim=keepdims == 1)
        ans.ub = op(abstracts[0].ub * multiplies, dim=axes, keepdim=keepdims == 1)

        ans.shape = abstracts[0].shape.copy()
        ans.splits = abstracts[0].splits.copy()
        if keepdims == 1:
            for d in axes:
                ans.shape[d] = 1
                ans.splits[d] = [0]
        else:
            for d in axes:
                ans.shape[d] = None
                ans.splits[d] = None

            ans.shape = list(filter(lambda x: x is not None, ans.shape))
            ans.splits = list(filter(lambda x: x is not None, ans.splits))

        ans.var_name = var_name
        return ans, list()

    def interp_ReduceMean(self, abstracts, node, optype, var_name, op=torch.sum):
        return self.interp_ReduceSum(abstracts, node, optype, var_name, op=torch.mean)

    def interp_ArgMax(self, abstracts, node, optype, var_name, mode='max'):
        attr = parse_attribute(node)
        axis = attr.get('axis', 0)
        keepdims = attr.get('keepdims', 1)
        select_last_index = attr.get('select_last_index', 0)

        data = abstracts[0]
        if mode == 'max':
            max_lb = torch.amax(data.lb, dim=axis, keepdim=True)
            possible = (data.ub >= max_lb)
        else:
            min_ub = torch.amin(data.ub, dim=axis, keepdim=True)
            possible = (data.lb <= min_ub)
        ans_lb = possible.max(axis, keepdim=bool(keepdims))[1]
        ans_ub = possible.shape[axis] - 1 - possible.flip(dims=[axis]).max(dim=axis, keepdim=bool(keepdims))[1]

        if len(data.splits[axis]) < data.shape[axis]:
            lb_mapping = torch.tensor(data.splits[axis], device=ans_lb.device, dtype=torch.float64)
            ub_mapping = torch.tensor(data.splits[axis][1:] + [data.shape[axis]], device=ans_lb.device, dtype=torch.float64) - 1
            ans_lb = lb_mapping[ans_lb]
            ans_ub = ub_mapping[ans_ub]

        ans = Abstraction()
        ans.lb = ans_lb
        ans.ub = ans_ub
        ans.splits = data.splits.copy()
        if keepdims:
            ans.splits[axis] = [0]
        else:
            del ans.splits[axis]
        ans.shape = data.shape.copy()
        if keepdims:
            ans.shape[axis] = 1
        else:
            del ans.shape[axis]
        ans.var_name = var_name
        return ans, list()

    def interp_ArgMin(self, abstracts, node, optype, var_name):
        return self.interp_ArgMax(abstracts, node, optype, var_name, mode='min')

    def interp_Tile(self, abstracts, node, optype, var_name):
        in_abst = abstracts[0]
        repeats = abstracts[1]
        assert repeats.is_exact()
        repeats = repeats.lb.detach().cpu().type(torch.int).tolist()
        if len(abstracts) > 2:
            axes = abstracts[2]
            assert axes.is_exact()
            axes = axes.lb.detach().cpu().type(torch.int).tolist()
        else:
            axes = list(range(in_abst.get_dim()))

        new_repeats = list()
        for new_axis in range(in_abst.get_dim()):
            old_ind = None
            if new_axis in axes:
                old_ind = axes.index(new_axis)
            if new_axis - in_abst.get_dim() in axes:
                old_ind = axes.index(new_axis - in_abst.get_dim())
            if old_ind is None:
                new_repeats.append(1)
            else:
                new_repeats.append(repeats[old_ind])

        ret = Abstraction()
        ret.lb = in_abst.lb.tile(tuple(new_repeats))
        ret.ub = in_abst.ub.tile(tuple(new_repeats))
        ret.shape = [item * new_repeats[i] for i, item in enumerate(in_abst.shape)]
        ret.splits = [[x for y in [[z + r * in_abst.shape[i] for z in split] for r in range(new_repeats[i])] for x in y]
                      for i, split in enumerate(in_abst.splits)]
        ret.var_name = var_name

        return ret, list()

    def interp_NegativeLogLikelihoodLoss(self, abstracts, node, optype, var_name):
        attrs = parse_attribute(node)
        ignore_index = attrs.get('ignore_index', None)
        reduction = attrs.get('reduction', b'mean')
        reduction = (reduction).decode('ascii')

        input = abstracts[0]
        target = abstracts[1]
        weight = abstracts[2] if len(abstracts) > 2 else None
        C = input.shape[1]
        exact_target = target.is_exact()

        # align input, target, and weight
        input = input.split_by([target.splits[0], input.splits[1] if weight is None else weight.splits[0]] + target.splits[1:], inplace=False)
        target = target.split_by([input.splits[0]] + input.splits[2:], inplace=False)
        if weight is not None:
            weight = weight.split_by([input.splits[1]], inplace=False)

        # get non-reduced values
        ans = Abstraction()
        if not exact_target:
            if weight is None:
                lb = input.lb
                ub = input.ub
            else:
                weight_lb_broadcast = weight.lb.view([1, -1] + [1] * (input.get_dim() - 2))
                weight_ub_broadcast = weight.ub.view([1, -1] + [1] * (input.get_dim() - 2))
                LL = input.lb * weight_lb_broadcast
                LU = input.lb * weight_ub_broadcast
                UL = input.ub * weight_lb_broadcast
                UU = input.ub * weight_ub_broadcast
                lb = torch.stack([LL, LU, UL, UU], dim=0).amin(dim=0)
                ub = torch.stack([LL, LU, UL, UU], dim=0).amax(dim=0)

            ans.lb = - ub.amax(dim=1)
            ans.ub = - lb.amin(dim=1)
            ans.shape = [input.shape[0]] + input.shape[2:]
            ans.splits = [input.splits[0]] + input.splits[2:]

            if ignore_index is not None:
                select_lb = torch.where((target.lb <= ignore_index) * (ignore_index <= target.ub),
                                        torch.zeros_like(ans.lb).to(ans.lb.device), ans.lb)
                select_ub = torch.where((target.lb <= ignore_index) * (ignore_index <= target.ub),
                                        torch.zeros_like(ans.ub).to(ans.ub.device), ans.ub)
                ans.lb = torch.minimum(ans.lb, select_lb)
                ans.ub = torch.maximum(ans.ub, select_ub)
        else:
            if weight is None:
                lb = input.lb
                ub = input.ub
            else:
                weight_lb_broadcast = weight.lb.view([1, -1] + [1] * (input.get_dim() - 2))
                weight_ub_broadcast = weight.ub.view([1, -1] + [1] * (input.get_dim() - 2))
                LL = input.lb * weight_lb_broadcast
                LU = input.lb * weight_ub_broadcast
                UL = input.ub * weight_lb_broadcast
                UU = input.ub * weight_ub_broadcast
                lb = torch.stack([LL, LU, UL, UU], dim=0).amin(dim=0)
                ub = torch.stack([LL, LU, UL, UU], dim=0).amax(dim=0)

            index_map = [bisect.bisect_right(input.splits[1], ind) - 1 for ind in range(C)]
            target_value = target.lb.long()

            if ignore_index is not None:
                # rewrite those equals to ignore_index to C
                # and map C to the new attached zero-valued slice (shape_1)
                lb = torch.cat([lb, torch.zeros([lb.shape[0], 1] + list(lb.shape[2:]))], dim=1)
                ub = torch.cat([ub, torch.zeros([ub.shape[0], 1] + list(ub.shape[2:]))], dim=1)
                target_value = torch.where(target_value == ignore_index,
                                           torch.ones_like(target_value).to(target_value.device).long() * len(input.splits[1]),
                                           target_value)
                index_map.append(lb.shape[1])

            index_map = torch.tensor(index_map).to(target_value.device)

            ans.lb = torch.nn.functional.nll_loss(ub, index_map[target_value], reduction='none')
            ans.ub = torch.nn.functional.nll_loss(lb, index_map[target_value], reduction='none')
            ans.shape = [input.shape[0]] + input.shape[2:]
            ans.splits = [input.splits[0]] + input.splits[2:]

        if reduction == 'mean':
            numel = reduce(lambda x, y: x * y, [input.shape[0]] + input.shape[2:])
            all_dim = list(range(ans.get_dim()))
            multiplies = self.cal_multiplies_for_sum(ans, all_dim)

            if ignore_index is None:
                possible_ignore_numels, certain_ignore_numels = 0, 0
            else:
                possible_ignore_numels = torch.sum((target.lb <= ignore_index) * (ignore_index <= target.ub) * multiplies)
                certain_ignore_numels = torch.sum((target.lb == ignore_index) * (ignore_index == target.ub) * multiplies)

            if weight is None:
                if ignore_index is None:
                    deno_min = deno_max = float(numel)
                else:
                    deno_min = float(numel - possible_ignore_numels)
                    deno_max = float(numel - certain_ignore_numels)
            else:
                if not exact_target:
                    w_min = torch.amin(weight.lb)
                    w_max = torch.amax(weight.ub)
                    deno_min = min(w_min * (numel - possible_ignore_numels), w_min * (numel - certain_ignore_numels))
                    deno_max = max(w_max * (numel - possible_ignore_numels), w_max * (numel - certain_ignore_numels))
                else:
                    weight_lb_broadcast = weight.lb.view([1, -1] + [1] * (input.get_dim() - 2))
                    weight_ub_broadcast = weight.ub.view([1, -1] + [1] * (input.get_dim() - 2))
                    wL = torch.ones_like(input.lb) * weight_lb_broadcast
                    wU = torch.ones_like(input.ub) * weight_ub_broadcast

                    index_map = [bisect.bisect_right(input.splits[1], ind) - 1 for ind in range(C)]
                    target_value = target.lb.long()

                    if ignore_index is not None:
                        # rewrite those equals to ignore_index to C
                        # and map C to the new attached zero-valued slice (shape_1)
                        wL = torch.cat([wL, torch.zeros([wL.shape[0], 1] + list(wL.shape[2:]))], dim=1)
                        wU = torch.cat([wU, torch.zeros([wU.shape[0], 1] + list(wU.shape[2:]))], dim=1)
                        target_value = torch.where(target_value == ignore_index,
                                                   torch.ones_like(target_value).to(target_value.device).long() * len(input.splits[1]),
                                                   target_value)
                        index_map.append(lb.shape[1])

                    index_map = torch.tensor(index_map).to(target_value.device)

                    wL = - torch.nn.functional.nll_loss(wL, index_map[target_value], reduction='none')
                    wU = - torch.nn.functional.nll_loss(wU, index_map[target_value], reduction='none')

                    wL, wU = wL * multiplies, wU * multiplies

                    deno_min = torch.sum(wL)
                    deno_max = torch.sum(wU)

        if reduction in ['sum', 'mean']:
            all_dim = list(range(ans.get_dim()))
            multiplies = self.cal_multiplies_for_sum(ans, all_dim)
            ans.lb = torch.sum(ans.lb * multiplies, dim=all_dim)
            ans.ub = torch.sum(ans.ub * multiplies, dim=all_dim)
            ans.shape = list()
            ans.splits = list()

        if reduction == 'mean':
            if deno_min <= PossibleNumericalError.UNDERFLOW_LIMIT and deno_max >= -PossibleNumericalError.UNDERFLOW_LIMIT:
                return None, [
                    PossibleNumericalError(optype, var_name, [torch.tensor(deno_min), torch.tensor(deno_max)],
                                           PossibleNumericalError.ERROR_CONTAINS_ZERO)]
            elif deno_max < 0.:
                ans.lb = torch.minimum(ans.ub / deno_min, ans.ub / deno_max)
                ans.ub = torch.maximum(ans.lb / deno_min, ans.lb / deno_max)
            else:
                # deno_min > 0.
                ans.lb = torch.minimum(ans.lb / deno_min, ans.lb / deno_max)
                ans.ub = torch.maximum(ans.ub / deno_min, ans.ub / deno_max)

        ans.var_name = var_name
        return ans, list()

    def interp_Loop(self, abstracts, node, optype, var_name, eps=1e-5, loop_dependencies=dict()):

        def _possibly_terminate(loop_i, trip_count, cond):
            # print(loop_i)
            # trip_count.print()
            # cond.print()
            if trip_count.is_empty():
                if cond.is_empty():
                    print('dead loop detected...')
                    return False
                else:
                    return (cond.lb.detach().cpu().item() <= 0. + eps)
            else:
                trip_count_lb = trip_count.lb.detach().cpu().item()
                if cond.is_empty():
                    return (loop_i >= trip_count_lb)
                else:
                    return (loop_i >= trip_count_lb or cond.lb.detach().cpu().item() <= 0. + eps)

        def _scan_output_concat(lst, name, device_for_empty):
            if len(lst) > 0:
                ans = Abstraction()
                ans.lb = torch.stack([item.lb for item in lst])
                ans.ub = torch.stack([item.ub for item in lst])
                ans.splits = [list(range(len(lst)))] + lst[0].splits
                ans.shape = [len(lst)] + lst[0].shape
                ans.name = name
            else:
                ans = create_empty_tensor(device_for_empty)
            return ans

        # print(abstracts)
        attr = parse_attribute(node)
        loop_body = attr['body']

        M = abstracts[0]
        cond = abstracts[1]
        v_initials = abstracts[2:]

        subgraph_abst_dict = loop_dependencies.copy()
        subgraph_scan_output_dict = dict()

        # init cond and out variables
        subgraph_abst_dict[loop_body.output[0].name] = cond
        for i, outp in enumerate(loop_body.output[1: 1 + len(v_initials)]):
            subgraph_abst_dict[outp.name] = v_initials[i]
        for i, outp in enumerate(loop_body.output[1 + len(v_initials):]):
            subgraph_scan_output_dict[outp.name] = list()

        possible_err = list()

        # start the loop and init trip count
        loop_i = 0
        conf_precise = AbstractionInitConfig(diff=False, from_init=True)
        subgraph_abst_dict[loop_body.input[0].name] = Abstraction().load(conf_precise, loop_body.input[0].name, [1],
                                                                         'INT', np.array(loop_i), M.lb.is_cuda)

        # print(f'loop @ {node.name}')

        while not _possibly_terminate(loop_i, M, cond):
            # carry condition to input cond
            subgraph_abst_dict[loop_body.input[1].name] = subgraph_abst_dict[loop_body.output[0].name]
            # short-cut loop-carried out vars to in vars
            for i, outp in enumerate(loop_body.output[1: 1 + len(v_initials)]):
                subgraph_abst_dict[loop_body.input[2 + i].name] = subgraph_abst_dict[outp.name]

            # execution
            possible_numerial_error_subgraph = self.subgraph_executor(subgraph_abst_dict, loop_body)
            possible_err.extend(possible_numerial_error_subgraph)

            # update new trip count
            loop_i += 1
            subgraph_abst_dict[loop_body.input[0].name] = Abstraction().load(conf_precise, loop_body.input[0].name, [1],
                                                                             'INT', np.array(loop_i), M.lb.is_cuda)

            # record current scan_outputs
            for outp in loop_body.output[1 + len(v_initials):]:
                subgraph_scan_output_dict[outp.name].append(subgraph_abst_dict[outp.name])

            # update condition
            cond = subgraph_abst_dict[loop_body.output[0].name]

        ret = [subgraph_abst_dict[x] for x in [y.name for y in loop_body.output[1: 1 + len(v_initials)]]] + \
              [_scan_output_concat(subgraph_scan_output_dict[k], k, M.lb.device)
               for k in [y.name for y in loop_body.output[1 + len(v_initials):]]]

        for x, y in zip(ret, node.output):
            x.var_name = y

        return ret, possible_err

    def interp_SequenceInsert(self, abstracts, node, optype, var_name):
        index = len(abstracts[0].lb)
        if len(abstracts) >= 3:
            assert abstracts[2].is_exact()
            index = int(abstracts[2].lb.detach().cpu().item())
        S = abstracts[0]
        T = abstracts[1]
        ret = Abstraction()
        ret.lb = S.lb.copy()
        ret.lb.insert(index, T.lb)
        ret.ub = S.ub.copy()
        ret.ub.insert(index, T.ub)
        ret.splits = S.splits.copy()
        ret.splits.insert(index, T.splits)
        ret.shape = S.shape.copy()
        ret.shape.insert(index, T.shape)
        ret.var_name = var_name
        return ret, list()

    def general_flatten(self, abstract: Abstraction, start_dim=0):
        t = start_dim
        for i in range(start_dim, len(abstract.shape)):
            if len(abstract.splits[i]) > 1:
                t = i

        flatten_orig_lb = abstract.lb.reshape(list(abstract.lb.shape[:start_dim]) + [-1])
        flatten_orig_ub = abstract.ub.reshape(list(abstract.ub.shape[:start_dim]) + [-1])

        abst_last_flat_dim = abstract.lb.shape[t]
        new_abst_last_dim = abst_last_flat_dim
        for i in range(start_dim, t):
            new_abst_last_dim *= abstract.shape[i]

        new_last_dim = 1
        for i in range(start_dim, len(abstract.shape)):
            new_last_dim *= abstract.shape[i]
        new_unit = 1
        for i in range(t, len(abstract.shape)):
            new_unit *= abstract.shape[i]

        indexes = list()
        for i in range(new_abst_last_dim):
            tmp = int(i / abst_last_flat_dim)
            orig_indexes = list()
            for now_dim in range(t - 1, start_dim - 1, -1):
                # (t-1), (t-2), ..., start_dim
                orig_indexes.append(tmp % abstract.shape[now_dim])
                tmp = int(tmp / abstract.shape[now_dim])
            orig_indexes = orig_indexes[::-1]
            # print(i, orig_indexes)
            # future work: optimize via binary search
            abst_indexes = [sum([now_ind >= x for x in abstract.splits[j + start_dim]]) - 1 for j, now_ind in
                            enumerate(orig_indexes)]
            abst_flatten_index = 0
            for j in range(start_dim, t):
                abst_flatten_index += abst_indexes[j - start_dim]
                abst_flatten_index *= abstract.lb.shape[j + 1]
            abst_flatten_index += (i % abst_last_flat_dim)
            indexes.append(abst_flatten_index)
            # print(i, abst_flatten_index)

        # print(t)
        # print(abst_last_flat_dim)
        # print(new_abst_last_dim)
        # print(new_last_dim)
        # print('new_unit', new_unit, abstract.shape[t])
        # print(abstract.splits[start_dim])
        # print(indexes)

        ans = Abstraction()
        ans.shape = abstract.shape[:start_dim] + [new_last_dim]
        ans.splits = abstract.splits[:start_dim] + \
                     [np.hstack([np.hstack([np.array(abstract.splits[t]) * int(new_unit / abstract.shape[t]) +
                                            time * new_unit for time in
                                            range(int(new_last_dim / new_unit))])]).tolist()]
        ans.lb = flatten_orig_lb.index_select(dim=start_dim, index=torch.tensor(indexes).to(flatten_orig_lb.device))
        ans.ub = flatten_orig_ub.index_select(dim=start_dim, index=torch.tensor(indexes).to(flatten_orig_ub.device))
        ans.var_name = abstract.var_name + f'_general_flatten_{start_dim}'

        return ans

    def general_stretch(self, abstract, start_dim, target_shape):
        assert start_dim == abstract.get_dim() - 1
        assert all([x == y for x, y in zip(abstract.shape[:start_dim], target_shape[:start_dim])])
        numels_to_stretch = 1
        for item in target_shape[start_dim:]:
            numels_to_stretch *= item
        assert numels_to_stretch == abstract.shape[start_dim]

        split_points = [list() for _ in target_shape[start_dim:]]
        for item in abstract.splits[start_dim]:
            for now_dim in range(len(target_shape) - 1, start_dim - 1, -1):
                split_points[now_dim - start_dim].append(item % target_shape[now_dim])
                item = int(item / target_shape[now_dim])
        split_points = [sorted(list(set(item))) for item in split_points]
        tot_numel = get_numel([len(x) for x in split_points])

        index_mapping = list()
        for i in range(tot_numel):
            cur_mapping = [None for _ in range(start_dim, len(target_shape))]
            for now_dim in range(len(target_shape) - 1, start_dim - 1, -1):
                cur_mapping[now_dim - start_dim] = i % len(split_points[now_dim - start_dim])
                i = int(i / len(split_points[now_dim - start_dim]))
            # print(i, cur_mapping)
            min_ind, max_ind = 0, 0
            for now_dim in range(start_dim, len(target_shape)):
                min_ind *= target_shape[now_dim]
                max_ind *= target_shape[now_dim]
                min_ind += split_points[now_dim - start_dim][cur_mapping[now_dim - start_dim]]
                if cur_mapping[now_dim - start_dim] == len(split_points[now_dim - start_dim]) - 1:
                    max_ind += target_shape[now_dim] - 1
                else:
                    max_ind += split_points[now_dim - start_dim][cur_mapping[now_dim - start_dim] + 1] - 1
            index_mapping.append((min_ind, max_ind))
        # print(index_mapping)
        tmp = list()
        for l, r in index_mapping:
            # future work: optimize via binary search
            real_index_l = max([i for i, item in enumerate(abstract.splits[start_dim]) if l >= item])
            real_index_r = min(
                [i for i, item in enumerate(abstract.splits[start_dim] + [abstract.shape[start_dim]]) if r < item]) - 1
            # print(real_index_l, real_index_r)
            assert real_index_l == real_index_r
            tmp.append(real_index_l)
        index_mapping = tmp

        ans = Abstraction()
        ans.splits = abstract.splits[:start_dim] + split_points
        ans.lb = abstract.lb.index_select(dim=start_dim, index=torch.tensor(index_mapping).to(abstract.lb.device))
        ans.ub = abstract.ub.index_select(dim=start_dim, index=torch.tensor(index_mapping).to(abstract.ub.device))
        ans.lb = ans.lb.reshape([len(x) for x in ans.splits])
        ans.ub = ans.ub.reshape([len(x) for x in ans.splits])
        ans.shape = target_shape
        ans.var_name = abstract.var_name + '_general_stretch'

        return ans

    def subgraph_executor(self, abst_dict: dict, subgraph: onnx.GraphProto):
        # print('input:', [y.name for y in subgraph.input])
        # print('output:', [y.name for y in subgraph.output])

        """ follow the same template as intep_module - construct graph, topology sort, and interp execution """
        """ the abstraction is updated in-place """

        # print('construct subgraph')
        deg_in = dict()
        deg_out = dict()
        # edge format:
        # key: vi, value: list
        # where each element of value corresponds to (vj, vi index in node input, vj index in node output, node name, node)
        edges = dict()
        all_nodes = set()

        for node in subgraph.node:
            if node.domain != '':
                print(f"  found domain def: {node.domain}")

            for v in list(node.input) + list(node.output):
                all_nodes.add(v)

            for i, vi in enumerate(list(node.input)):
                for j, vj in enumerate(list(node.output)):
                    if vi not in edges: edges[vi] = list()
                    edges[vi].append((vj, i, j, node.op_type, node.name, node))

                    if vi not in deg_out:
                        deg_out[vi] = 1
                    else:
                        deg_out[vi] += 1
                    if vj not in deg_in:
                        deg_in[vj] = 1
                    else:
                        deg_in[vj] += 1

            # new component: handle Constant node
            if len(node.input) == 0:
                if node.op_type != 'Constant':
                    print(f'  warning: in sub-graph, detect another optype for constant node: {node.op_type}')
                attr = parse_attribute(node)
                value = numpy_helper.to_array(attr['value'])
                for v in abst_dict.values():
                    is_cuda = v.lb.is_cuda
                    break
                abst_dict[node.output[0]] = Abstraction().load(self.loop_constant_abst_cfg,
                                                               node.output[0],
                                                               value.shape,
                                                               datatype_mapping[attr['value'].data_type],
                                                               value, is_cuda)

        for v in all_nodes:
            if v not in deg_in: deg_in[v] = 0
            if v not in deg_out: deg_out[v] = 0
            if v not in edges: edges[v] = list()

        possible_numerical_errors = list()

        start_points = set([x for x in all_nodes if deg_in[x] == 0])
        refresh_nodes_in_this_round = start_points
        queue = list(start_points)
        cur_deg_in = deg_in.copy()

        # print('subgraph topology sort')
        l = 0
        while l < len(queue):
            cur_var = queue[l]
            for vj, ind_i, ind_j, node_optype, node_name, node in edges[cur_var]:
                cur_deg_in[vj] -= 1
                if cur_deg_in[vj] == 0:
                    if vj in refresh_nodes_in_this_round:
                        # already obtained the abstraction from previous runs,
                        # only happens for nodes with multiple outputs,
                        # so now we can skip
                        queue.append(vj)
                        pass

                    else:
                        cur_abst, cur_exceps = self.handle(
                            [abst_dict[x] for x in node.input], node, node_optype, vj
                        )
                        possible_numerical_errors.extend(cur_exceps)
                        if cur_abst is None:
                            print(f'! No abstraction generated for {vj}: '
                                  f'node name = {node_name}, type = {node_optype}')
                        else:
                            queue.append(vj)
                            if isinstance(cur_abst, Abstraction):
                                # single output node
                                abst_dict[vj] = cur_abst
                                refresh_nodes_in_this_round.add(vj)
                                # print(vj, 'updated to:')
                                # cur_abst.print()
                            else:
                                # multiple output node: execute once, update all output nodes
                                for i, cur_cur_abst in enumerate(cur_abst):
                                    abst_dict[node.output[i]] = cur_cur_abst
                                    refresh_nodes_in_this_round.add(node.output[i])
            l += 1

        return possible_numerical_errors

    def cal_multiplies_for_sum(self, abstract, axes: list):
        # compute multiplies to calculate reduced sum
        multiplies = torch.ones_like(abstract.lb)
        for d in axes:
            splits = abstract.splits[d] + [abstract.shape[d]]
            split_sizes = [splits[i] - splits[i - 1] for i in range(1, len(splits))]
            view_shape = [1] * len(abstract.shape)
            view_shape[d] = len(abstract.splits[d])
            multiplies *= torch.tensor(split_sizes).view(view_shape)

        return multiplies


def get_shape_split_with_broadcasting(a: Abstraction, b: Abstraction):
    """
        Generating the shape and splits information after the broadcasting-supported operation of two tensors,
            where broadcasting singleton dimension could be possible
    :param a:
    :param b:
    :return:
    """
    shape = list()
    splits = list()
    assert a.get_dim() == b.get_dim()
    for i in range(a.get_dim()):
        if a.shape[i] >= b.shape[i]:
            shape.append(a.shape[i])
            splits.append(a.splits[i])
        else:
            shape.append(b.shape[i])
            splits.append(b.splits[i])
    return shape, splits


def create_empty_tensor(device='cpu'):
    """
        Create an empty tensor
        the empty tensor needs to have shape = [0], to distinguish from a scalar tensor whose shape is []
    :param device:
        an update: set the default device to "cpu"
    :return:
    """
    ret = Abstraction()
    ret.var_name = 'null'
    ret.shape = [0]
    ret.splits = [[]]
    ret.lb = torch.tensor([], device=device)
    ret.ub = torch.tensor([], device=device)
    return ret


def compute_outshape_padding(pad_mode, prev_pads, x_conv_shape, kernel_shape, dilations, strides, ceil_mode=False):
    dim_for_conv = len(x_conv_shape)
    if prev_pads is not None:
        prev_pads_order_by_dim = []
        out_shape = []
        for i in range(dim_for_conv):
            prev_pads_order_by_dim.extend([prev_pads[i], prev_pads[i + dim_for_conv]])
            padding_size = prev_pads[i] + prev_pads[i + dim_for_conv]
            if ceil_mode:
                out_shape.append(math.ceil(
                    (x_conv_shape[i] + padding_size - ((kernel_shape[i] - 1) * dilations[i] + 1)) / strides[i] + 1))
            else:
                out_shape.append(math.floor(
                    (x_conv_shape[i] + padding_size - ((kernel_shape[i] - 1) * dilations[i] + 1)) / strides[i] + 1))
        return out_shape, prev_pads_order_by_dim
    else:
        if not isinstance(pad_mode, str):
            pad_mode = pad_mode.decode('ascii')
        padding_deltas = []
        out_shape = []
        if pad_mode in ["VALID", "NOTSET"]:
            for i in range(dim_for_conv):
                out_shape.append(math.ceil(
                    (x_conv_shape[i] - ((kernel_shape[i] - 1) * dilations[i] + 1) + 1) /
                    strides[i]))
                padding_deltas.extend([0, (out_shape[i] - 1) * strides[i] + (
                        (kernel_shape[i] - 1) * dilations[i] + 1) - x_conv_shape[i]])

        elif pad_mode in ['SAME_UPPER', 'SAME_LOWER']:
            for i in range(dim_for_conv):
                out_shape.append(math.ceil(x_conv_shape[i] / strides[i]))
                delta = (out_shape[i] - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - x_conv_shape[i]
                if pad_mode == 'SAME_UPPER':
                    padding_deltas.extend([delta // 2, delta - delta // 2])
                else:
                    padding_deltas.extend([delta - delta // 2, delta // 2])

        return out_shape, padding_deltas


def compute_outshape_padding_trans(pad_mode, prev_pads, output_shape, output_padding,
                                   x_conv_shape, kernel_shape, dilations, strides):
    # note: the padding is on output and is indeed cropping instead of padding
    dim_for_conv = len(x_conv_shape)

    if not isinstance(pad_mode, str):
        pad_mode = pad_mode.decode('ascii')

    if output_shape is not None:
        out_shape = output_shape.copy()
        padding = list()
        for i in range(dim_for_conv):
            total_padding = strides[i] * (x_conv_shape[i] - 1) + ((kernel_shape[i] - 1) * dilations[i] + 1) - out_shape[i]
            if pad_mode == 'SAME_UPPER':
                padding.extend([total_padding // 2, total_padding - total_padding // 2])
            else:
                padding.extend([total_padding - total_padding // 2, total_padding // 2])
        return out_shape, padding
    elif prev_pads is not None:
        prev_pads_order_by_dim = []
        out_shape = list()
        for i in range(dim_for_conv):
            low, high = prev_pads[i], prev_pads[i + dim_for_conv]
            if output_padding is not None:
                high -= output_padding[i]
            prev_pads_order_by_dim.extend([low, high])
            padding_size = low + high
            out_shape.append(strides[i] * (x_conv_shape[i] - 1) + ((kernel_shape[i] - 1) * dilations[i] + 1) - padding_size)
        return out_shape, prev_pads_order_by_dim
    else:
        out_shape = list()
        padding_deltas = list()
        if pad_mode in ["VALID", "NOTSET"]:
            for i in range(dim_for_conv):
                out_shape.append(strides[i] * (x_conv_shape[i] - 1) + ((kernel_shape[i] - 1) * dilations[i] + 1))
                padding_deltas.extend([0, 0])
                if output_padding is not None:
                    out_shape[-1] += output_padding[i]
                    padding_deltas[-1] -= output_padding[i]
        elif pad_mode in ['SAME_UPPER', 'SAME_LOWER']:
            for i in range(dim_for_conv):
                out_shape.append(strides[i] * x_conv_shape[i])
                total_padding = strides[i] * (x_conv_shape[i] - 1) + ((kernel_shape[i] - 1) * dilations[i] + 1) - out_shape[-1]
                if pad_mode == 'SAME_UPPER':
                    padding_deltas.extend([total_padding // 2, total_padding - total_padding // 2])
                else:
                    padding_deltas.extend([total_padding - total_padding // 2, total_padding // 2])
                if output_padding is not None:
                    padding_deltas[-1] -= output_padding[-1]
                    out_shape[-1] += output_padding[i]

        return out_shape, padding_deltas


def add_padding_to_X(X, padding, value, mode='value', shift=2, value_ub=None):
    assert mode in ['value', 'minmax', 'edge', 'reflect']
    if any(padding < 0):
        # trim the input if there exists padding < 0
        # print('detected padding < 0 case')
        for i in range(len(padding)):
            if padding[i] < 0:
                begin = i % 2 == 0
                index_of_padding = i // 2
                if begin:
                    start_h_ind = bisect.bisect_right(X.splits[shift + index_of_padding], -padding[i]) - 1
                    if start_h_ind > 0:
                        indexes = [slice(None)] * len(X.shape)
                        indexes[shift + index_of_padding] = slice(start_h_ind, None)
                        # print('start', indexes, X.lb.shape)
                        X.lb = X.lb[indexes]
                        # print(X.lb.shape)
                        X.ub = X.ub[indexes]
                        X.splits[shift + index_of_padding] = [max(item + padding[i], 0) for item in
                                                          X.splits[shift + index_of_padding][start_h_ind:]]
                else:
                    end_h_ind = bisect.bisect_right(X.splits[shift + index_of_padding],
                                                    X.shape[shift + index_of_padding] + padding[i] - 1)
                    if end_h_ind < X.lb.shape[shift + index_of_padding]:
                        indexes = [slice(None)] * len(X.shape)
                        indexes[shift + index_of_padding] = slice(None, end_h_ind)
                        # print('end', indexes, X.lb.shape)
                        X.lb = X.lb[indexes]
                        # print(X.lb.shape)
                        X.ub = X.ub[indexes]
                        X.splits[shift + index_of_padding] = X.splits[shift + index_of_padding][:end_h_ind]

                X.shape[2 + index_of_padding] += padding[i]

    elif any(padding > 0):
        # prepend the input if there exists padding > 0
        if mode != 'reflect':
            for i in range(len(padding)):
                if padding[i] > 0:
                    begin = i % 2 == 0
                    index_of_padding = i // 2
                    if mode in ['value', 'minmax']:
                        to_pend_shape = list(X.lb.shape)
                        to_pend_shape[shift + index_of_padding] = 1
                        if mode == 'value':
                            to_pend_min = torch.full(to_pend_shape, value).to(X.lb.device)
                            if value_ub is None:
                                to_pend_max = to_pend_min
                            else:
                                to_pend_max = torch.full(to_pend_shape, value_ub).to(X.ub.device)
                        elif mode == 'minmax':
                            to_pend_min = torch.amin(X.lb, dim=shift + index_of_padding, keepdim=True)
                            to_pend_max = torch.amax(X.ub, dim=shift + index_of_padding, keepdim=True)
                        if begin:
                            X.lb = torch.cat([to_pend_min, X.lb], dim=shift + index_of_padding)
                            X.ub = torch.cat([to_pend_max, X.ub], dim=shift + index_of_padding)
                            X.splits[shift + index_of_padding] = [0] + [item + padding[i] for item in
                                                                    X.splits[shift + index_of_padding]]
                            X.shape[shift + index_of_padding] += padding[i]
                        else:
                            X.lb = torch.cat([X.lb, to_pend_min], dim=shift + index_of_padding)
                            X.ub = torch.cat([X.ub, to_pend_max], dim=shift + index_of_padding)
                            X.splits[shift + index_of_padding] = X.splits[shift + index_of_padding] + [
                                X.shape[shift + index_of_padding]]
                            X.shape[shift + index_of_padding] += padding[i]
                    elif mode == 'edge':
                        X.shape[shift + index_of_padding] += padding[i]
                        if begin:
                            X.splits[shift + index_of_padding] = [X.splits[shift + index_of_padding][0]] + \
                                                                 [item + padding[i] for item in X.splits[shift + index_of_padding][1:]]
        else:
            for i in range(len(padding) // 2):
                begin = padding[2 * i]
                end = padding[2 * i + 1]
                if begin > 0 or end > 0:
                    ref_splits = X.splits.copy()
                    ref_splits[shift + i] = set(range(min(begin + 1, X.shape[shift + i]))).union(set(range(max(X.shape[shift + i]-1-end,0), X.shape[shift + i])))
                    ref_splits[shift + i] = sorted(list(ref_splits[shift + i]))
                    X.split_by(ref_splits, inplace=True)
                    lst_lb, lst_ub = list(), list()
                    if begin > 0:
                        lb_prepend = torch.index_select(X.lb, shift + i, torch.tensor([j % X.shape[shift + i] for j in range(begin, 0, -1)], dtype=torch.long, device=X.lb.device))
                        ub_prepend = torch.index_select(X.ub, shift + i, torch.tensor([j % X.shape[shift + i] for j in range(begin, 0, -1)], dtype=torch.long, device=X.ub.device))
                        lst_lb.append(lb_prepend)
                        lst_ub.append(ub_prepend)
                    lst_lb.append(X.lb)
                    lst_ub.append(X.ub)
                    if end > 0:
                        lb_append = torch.index_select(X.lb, shift + i, torch.tensor([j % X.shape[shift + i] for j in range(X.shape[shift + i]-2, X.shape[shift + i]-end-2, -1)], dtype=torch.long, device=X.lb.device))
                        ub_append = torch.index_select(X.ub, shift + i, torch.tensor([j % X.shape[shift + i] for j in range(X.shape[shift + i]-2, X.shape[shift + i]-end-2, -1)], dtype=torch.long, device=X.lb.device))
                        lst_lb.append(lb_append)
                        lst_ub.append(ub_append)
                    X.lb = torch.cat(lst_lb, dim=shift + i)
                    X.ub = torch.cat(lst_ub, dim=shift + i)
                    X.splits[shift + i] = list(range(begin)) + [item + begin for item in X.splits[shift + i]] + list(range(X.shape[shift + i] + begin, X.shape[shift + i] + begin + end))
                    X.shape[shift + i] += begin + end

def fold_repeated_indices(X, dim_for_conv, out_shape, kernel_shape, dilations, strides):
    lpses = []  # the list of [lpxs, lpys] when dim_for_conv == 2
    rpses = []  # the list of [rpxs, rpys] when dim_for_conv == 2
    repses = []  # the list of [xreps, yreps] when dim_for_conv == 2
    for i in range(dim_for_conv):
        lp = rp = 0
        lx = 0
        rx = dilations[i] * (kernel_shape[i] - 1)
        lpxs, rpxs = list(), list()
        xreps = [0 for _ in X.splits[2 + i]]

        for x in range(out_shape[i]):
            prelp = lp
            while lp < X.lb.shape[2 + i] - 1 and X.splits[2 + i][lp + 1] <= lx:
                lp += 1
            while rp < X.lb.shape[2 + i] - 1 and X.splits[2 + i][rp + 1] <= rx:
                rp += 1
            # print(lx, rx, lp, rp)
            if lp == rp == prelp and lx != 0:
                xreps[lp] += 1
            lpxs.append(lp)
            rpxs.append(rp)
            lx += strides[i]
            rx += strides[i]

        lpses.append(lpxs)
        rpses.append(rpxs)
        repses.append(xreps)

    # print(repses)

    """fold repeated indices"""
    blklens = [np.array(X.splits[2 + i][1:] + [X.shape[2 + i]]) - np.array(X.splits[2 + i]) for i in
               range(dim_for_conv)]
    indexes = [[i for i, lx in enumerate(X.splits[2 + index])
                for _ in range(blklens[index][i] - repses[index][i] * strides[index])] for index
               in range(dim_for_conv)]

    # print(x_index, y_index)
    # print(len(x_index), len(y_index))

    new_X_lb = X.lb
    new_X_ub = X.ub
    for i in range(dim_for_conv):
        new_X_lb = new_X_lb.index_select(dim=2 + i, index=torch.tensor(indexes[i]).to(X.lb.device))
        new_X_ub = new_X_ub.index_select(dim=2 + i, index=torch.tensor(indexes[i]).to(X.ub.device))

    return new_X_lb, new_X_ub, lpses, rpses


def fold_repeated_indices_trans(X, dim_for_convtrans, kernel_shape, dilations, strides):
    lpses = []  # the list of [lpxs, lpys] when dim_for_convtrans == 2
    rpses = []  # the list of [rpxs, rpys] when dim_for_convtrans == 2
    repses = []  # the list of [xreps, yreps] when dim_for_convtrans == 2

    for i in range(dim_for_convtrans):
        p = 0
        prev_avail = -1

        lpxs, rpxs, xreps = list(), list(), list()

        for x in range(X.shape[2 + i]):
            while p < X.lb.shape[2 + i] - 1 and X.splits[2 + i][p + 1] <= x:
                p += 1
            lx = strides[i] * x
            rx = strides[i] * x + (kernel_shape[i] - 1) * dilations[i]
            r_bound = (strides[i] * (X.splits[2 + i][p] - 1) + (kernel_shape[i] - 1) * dilations[i]) if p > 0 else \
                (strides[i] * (-1) + (kernel_shape[i] - 1) * dilations[i])
            l_bound = (strides[i] * (X.splits[2 + i][p + 1])) if p < X.lb.shape[2 + i] - 1 else \
                (strides[i] * X.shape[2 + i])
            if lx > r_bound and rx < l_bound:
                avail = p
            else:
                avail = -1
            if avail == prev_avail and avail != -1:
                rpxs[-1] = x
                xreps[-1] += 1
            else:
                lpxs.append(x)
                rpxs.append(x)
                xreps.append(1)
            prev_avail = avail

        lpses.append(lpxs)
        rpses.append(rpxs)
        repses.append(xreps)

    """fold repeated indices"""
    origindexes = [[j for j, times in enumerate(np.array(X.splits[2 + i][1:] + [X.shape[2 + i]]) - np.array(X.splits[2 + i])) for _ in range(times)]
               for i in range(dim_for_convtrans)]
    indexes = [[origindexes[index][item] for item in lpses[index]] for index
               in range(dim_for_convtrans)]

    new_X_lb = X.lb
    new_X_ub = X.ub
    for i in range(dim_for_convtrans):
        new_X_lb = new_X_lb.index_select(dim=2 + i, index=torch.tensor(indexes[i], dtype=torch.long).to(X.lb.device))
        new_X_ub = new_X_ub.index_select(dim=2 + i, index=torch.tensor(indexes[i], dtype=torch.long).to(X.lb.device))

    return new_X_lb, new_X_ub, lpses, rpses