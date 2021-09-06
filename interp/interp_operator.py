import numpy as np
import torch
import bisect

import onnx
from interp.interp_utils import AbstractionInitConfig, parse_attribute, unsupported_types, datatype_mapping, get_numel, \
    PossibleNumericalError


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
             tensor_data: None or np.ndarray,
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
                    raise Exception(f'Variable {var_name}: stride config {stride} should much {tensor_shape}')

            if len(tensor_shape) == 0:
                # scalar
                if from_init and tensor_data is not None:
                    tensor_data = tensor_data.reshape(())
                    self.lb, self.ub = \
                        torch.tensor(tensor_data - from_init_margin, dtype=torch.float32, requires_grad=diff), \
                        torch.tensor(tensor_data + from_init_margin, dtype=torch.float32, requires_grad=diff)
                else:
                    self.lb, self.ub = \
                        torch.tensor(lb, dtype=torch.float32, requires_grad=diff), \
                        torch.tensor(ub, dtype=torch.float32, requires_grad=diff)
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
                    lb_data = np.array(lb_data, dtype=np.float32) - from_init_margin
                    ub_data = np.array(ub_data, dtype=np.float32) + from_init_margin
                    self.lb, self.ub = \
                        torch.tensor(lb_data, dtype=torch.float32, requires_grad=diff), \
                        torch.tensor(ub_data, dtype=torch.float32, requires_grad=diff)

                else:
                    self.lb, self.ub = \
                        torch.tensor(lb * np.ones(abst_shape), dtype=torch.float32, requires_grad=diff), \
                        torch.tensor(ub * np.ones(abst_shape), dtype=torch.float32, requires_grad=diff)

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

        lb = torch.min(self.lb)
        ub = torch.max(self.ub)
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
                    if len(new_s) == 0 or ref_s[p2] > new_s[-1]:
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
                    now_lb.index_select(dim=dim, index=torch.tensor(range(old_l, old_r)).to(now_lb.device)).min(
                        dim=dim)[0])
                ts_list_ub.append(
                    now_ub.index_select(dim=dim, index=torch.tensor(range(old_l, old_r)).to(now_ub.device)).max(
                        dim=dim)[0])

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
        local_lb = self.lb.detach().reshape(-1)
        local_ub = self.ub.detach().reshape(-1)
        dif = torch.norm(local_ub - local_lb).item()
        return dif < eps

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
        print('lb_tensor shape:', self.lb.shape)
        print('ub_tensor:', self.ub)
        print('ub_tensor shape:', self.ub.shape)
        print('splits:', self.splits)
        print('shape:', self.shape)
        print('===ABST PRINT END===')

    def get_dim(self):
        return len(self.shape)


class Interpreter(object):
    """
        The general class for generating interpretations
    """

    def __init__(self, smash_thres=-1, ceil='precise', floor='precise'):
        # default smash threshold
        self.smash = smash_thres

        # whether to propagate precise or coarse bound for ceil or floor
        # the precise means that, ceil/floor exactly applies ceil/floor func, but this way we cannot have the gradient
        # the idential means that, we just propagate the identical value as the approximation and we can obtain the gradient
        # the coarse means that, the corase bound is (lb, ub+1) for ceil and (lb-1, ub) for floor, which is imprecise but we can obtain the gradient
        assert ceil in ['precise', 'identical', 'coarse']
        assert floor in ['precise', 'identical', 'coarse']
        self.ceil = ceil
        self.floor = floor

    def handle(self, abstracts, node, optype, var_name):
        """
            The dispatcher
        :param abstracts:
        :param node:
        :param var_name:
        :return: Abstraction instance, and list of exceptions
        """

        try:
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
        ans.lb = torch.min(choices, dim=0)[0]
        ans.ub = torch.max(choices, dim=0)[0]
        ans.var_name = var_name

        return ans, list()

    def interp_Div(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)
        if ((abst1.lb <= 0) & (abst1.ub >= 0)).any():
            return None, [
                PossibleNumericalError(optype, var_name, [abst1.lb, abst1.ub],
                                       PossibleNumericalError.ERROR_CONTAINS_ZERO)]

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        choices = torch.stack([abst0.lb / abst1.lb, abst0.lb / abst1.ub, abst0.ub / abst1.lb, abst0.ub / abst1.ub],
                              dim=0)
        ans.lb = torch.min(choices, dim=0)[0]
        ans.ub = torch.max(choices, dim=0)[0]
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

    def interp_Exp(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        if (abst.ub >= PossibleNumericalError.OVERFLOW_D * np.log(10)).any():
            return None, [
                PossibleNumericalError(optype, var_name, [abst.lb, abst.ub], PossibleNumericalError.ERROR_OVERFLOW)]
        ans.lb = torch.exp(abst.lb)
        ans.ub = torch.exp(abst.ub)
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
        # Todo a more tight lower/upper bound exists
        # inputs: [l1, l2, l3], [u1, u2, u3]
        # softmax_lb = [l1 / (l1 + u2 + u3), ...]
        # softmax_ub = [u1 / (u1 + l2 + l3)]
        ans.lb = exp_lb / torch.sum(exp_ub * multiplies, dim=axis, keepdim=True)
        ans.ub = exp_ub / torch.sum(exp_lb * multiplies, dim=axis, keepdim=True)

        ans.shape = abst.shape.copy()
        ans.splits = abst.splits.copy()
        return ans, list()

    def interp_Abs(self, abstracts, node, optype, var_name):
        abst = abstracts[0]
        ans = Abstraction()
        ans.lb = torch.where(abst.lb > 0, abst.lb, torch.zeros_like(abst.lb))
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

        assert abst_shape.is_exact()
        desired_shape = abst_shape.lb.detach().type(torch.int).tolist()

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
            starts = starts.lb.detach().cpu().type(torch.int32).tolist()
            assert ends.is_exact()
            ends = ends.lb.detach().cpu().type(torch.int32).tolist()
            if len(abstracts) >= 4:
                axes = abstracts[3]
                assert axes.is_exact()
                axes = axes.lb.detach().cpu().type(torch.int32).tolist()
            else:
                axes = list(range(len(starts)))
            if len(abstracts) >= 5:
                steps = abstracts[4]
                assert steps.is_exact()
                steps = steps.lb.detach().cpu().type(torch.int32).tolist()
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

            # print(now_axis, now_start, now_end)

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
        return now_abst, list()

    def interp_Squeeze(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        if 'axes' in attr:
            axes = attr['axes']
        elif len(abstracts) > 1:
            axes = abstracts[1]
            assert axes.is_exact()
            axes = axes.lb.detach().cpu().type(torch.int32).tolist()
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
            axes = axes.lb.detach().cpu().type(torch.int32).tolist()
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

    def interp_ConstantOfShape(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        value = attr.get('value', 0)
        device = abstracts[0].lb.device

        if abstracts[0].is_exact():
            ans = Abstraction()
            ans.shape = list(abstracts[0].lb.long().numpy())
            ans.splits = [[0] for _ in range(len(ans.shape))]
            ans.lb = torch.full([1] * len(ans.shape), value, device=device)
            ans.ub = torch.full([1] * len(ans.shape), value, device=device)
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
        ans.splits = abstracts[0].splits.copy()
        ans.lb = torch.full([1] * len(ans.shape), low, device=device)
        ans.ub = torch.full([1] * len(ans.shape), high, device=device)
        ans.var_name = var_name
        return ans, list()

    def interp_Less(self, abstracts, node, optype, var_name):
        abst0 = abstracts[0].extend_dim(abstracts[1].get_dim(), inplace=False)
        abst1 = abstracts[1].extend_dim(abstracts[0].get_dim(), inplace=False)

        abst0.split_by(abst1.splits, inplace=True)
        abst1.split_by(abst0.splits, inplace=True)

        ans = Abstraction()
        ans.shape, ans.splits = get_shape_split_with_broadcasting(abst0, abst1)
        ones = torch.ones_like(abst0.lb)
        zeros = torch.zeros_like(abst0.lb)
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
        ones = torch.ones_like(abst0.lb)
        zeros = torch.zeros_like(abst0.lb)
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
        ones = torch.ones_like(abst0.lb)
        zeros = torch.zeros_like(abst0.lb)
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
        ones = torch.ones_like(abst0.lb)
        zeros = torch.zeros_like(abst0.lb)
        ans.lb = torch.where(abst0.lb < abst1.ub, zeros, ones)
        ans.ub = torch.where(abst0.ub >= abst1.lb, ones, zeros)
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

    def interp_ReduceMin(self, abstracts, node, optype, var_name, op=torch.min):
        attr = parse_attribute(node)
        keepdims = attr.get('keepdims', 1)
        axes = attr.get('axes', None)
        assert axes is not None
        axes = [(axis + 3) % 3 for axis in axes]
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

    def interp_ReduceSum(self, abstracts, node, optype, var_name):
        attr = parse_attribute(node)
        keepdims = attr.get('keepdims', 1)
        axes = attr.get('axes', None)
        assert axes is not None
        ans = Abstraction()
        # compute multiplies to calculate reduced sum
        multiplies = self.cal_multiplies_for_sum(abstracts[0], axes)

        ans.lb = torch.sum(abstracts[0].lb * multiplies, dim=axes, keepdim=keepdims == 1)
        ans.ub = torch.sum(abstracts[0].ub * multiplies, dim=axes, keepdim=keepdims == 1)

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

    def interp_Loop(self, abstracts, node, optype, var_name):
        print(abstracts)
        attr = parse_attribute(node)
        loop_body = attr['body']
        print(loop_body.input)
        print(loop_body.output)

        return None, list()

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

    def cal_multiplies_for_sum(self, abstract, axes: list):
        # compute multiplies to calculate reduced sum
        multiplies = torch.ones_like(abstract.lb)
        for d in axes:
            splits = abstract.splits[d] + [abstract.shape[d]]
            split_sizes = [splits[i] - splits[i - 1] for i in range(1, len(splits))]
            view_shape = [1] * len(abstract.shape)
            view_shape[d] = len(abstract.splits[d])
            multiplies *= torch.Tensor(split_sizes).view(view_shape)

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
