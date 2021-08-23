
import numpy as np
import torch

from interp.interp_utils import AbstractionInitConfig

class Abstraction(object):
    """
        The Abstraction class is attached to each variable during the interpretation process
    """

    def __init__(self,
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
                except:
                    raise Exception(f'Variable {var_name}: tensor data (shape:{tensor_data.shape}) cannot be casted to required shape({tensor_shape})')

                lb_data, ub_data = self.split_and_assign(tensor_data, self.splits)
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


    @staticmethod
    def split_and_assign(tensor_data, splits, dim=0):
        # print(f'split and assign {tensor_data} {splits} dim={dim}')
        if dim == len(splits):
            return np.min(tensor_data), np.max(tensor_data)
        else:
            s_tensor_data = np.split(tensor_data, splits[dim][1:], axis=dim)
            # print(s_tensor_data)
            res = [Abstraction.split_and_assign(block, splits, dim + 1) for block in s_tensor_data]
            return [item[0] for item in res], [item[1] for item in res]

    def print(self):
        print('===ABST PRINT BEGIN===')
        print('var_name:', self.var_name)
        print('lb_tensor:', self.lb)
        print('lb_tensor shape:', self.lb.shape)
        print('ub_tensor:', self.ub)
        print('ub_tensor shape:', self.ub.shape)
        print('splits:', self.splits)
        print('===ABST PRINT END===')
