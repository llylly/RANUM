
import numpy as np
import torch

from interp.interp_utils import AbstractionInitConfig

class Abstraction(object):
    """
        The Abstraction class is attached to each variable during the interpretation process
    """

    def __init__(self):
        self.lb = self.ub = self.var_name = self.shape = self.splits = None

    def load(self,
             config: AbstractionInitConfig,
             var_name: str,
             tensor_shape: list,
             tensor_type: str,
             tensor_data: None or np.ndarray):

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
            for i,shape_i in enumerate(tensor_shape):
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
                    raise Exception(f'Variable {var_name}: tensor data (shape:{tensor_data.shape}) cannot be casted to required shape({tensor_shape})')

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

        self.var_name = var_name
        self.shape = tensor_shape

    def split_by(self, ref_splits, inplace=True):
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
                            new_index.append(p1-1)
                    p2 += 1
            # print(new_s)
            # print(new_index)

            new_lb = torch.index_select(new_lb, i, torch.tensor(new_index))
            new_ub = torch.index_select(new_ub, i, torch.tensor(new_index))
            new_splits.append(new_index)

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


    @staticmethod
    def summarize_data_and_assign(tensor_data, splits, dim=0):
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


class Interpreter(object):
    """
        The general class for generating interpretations
    """

    def __init__(self):
        pass

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
        abst0 = abstracts[0].split_by(abstracts[1].splits, inplace=False)
        abst1 = abstracts[1].split_by(abstracts[0].splits, inplace=False)

        ans = Abstraction()
        ans.shape = abst0.shape
        ans.splits = abst0.splits
        ans.lb = abst0.lb - abst1.ub
        ans.ub = abst0.ub - abst1.lb
        ans.var_name = var_name

        return ans, list()

    def interp_MatMul(self, abstracts, node, optype, var_name):

        for i, item in enumerate(abstracts):
            print(f'input #{i}')
            item.print()

        return None, list()

    def interp_Reshape(self, abstracts, node, optype, var_name):
        return None, list()
