import onnx
import torch
from onnx.helper import get_attribute_value
import torch

EPS = 1e-5

POSTIVE_MINIMUM = 1e-10


class AbstractionInitConfig(object):
    INPUT_CONFIG_DEFAULT = [-50., 50.]
    WEIGHT_CONFIG_DEFAULT = [-10., 10.]
    DROPOUT_CONFIG_DEFAULT = [0.1, 0,9]
    KEEP_PROB_CONFIG_DEFAULT = [0.1, 0.9]
    VARIANCE_CONFIG_DEFAULT = [0, 1]
    INT_CONFIG_DEFAULT = [0, 20]

    def __init__(self, diff: bool, lb=None, ub=None, from_init=False, from_init_margin=0., stride=-1):
        if lb is None:
            lb = AbstractionInitConfig.INPUT_CONFIG_DEFAULT[0]
        if ub is None:
            ub = AbstractionInitConfig.INPUT_CONFIG_DEFAULT[1]
        super(AbstractionInitConfig, self).__init__()
        # diff: whether requires_grad differentiable
        self.diff = diff
        # default lower bound of tensor values
        self.lb = lb
        # default upper bound of tensor values
        self.ub = ub
        # if real tensor values are present, whether to use them to determine lower and upper bounds
        self.from_init = from_init
        # if from_init=True, the relative margin of abstraction from the init data, set to 0 means precisely capture the init data's range
        self.from_init_margin = from_init_margin
        # the stride of abstract, int or list of int (if it is not a list, then cast to all dimensions)
        # "-1" means stride = infty, i.e., the whole dimension is abstracted by a single element
        self.stride = stride

    def load_from_dict(self, d):
        self.diff = bool(d['diff'])
        self.lb = float(d['lb'])
        self.ub = float(d['ub'])
        self.from_init = bool(d['from_init'])
        self.from_init_margin = float(d['from_init_margin'])
        self.stride = d['stride']
        return self

    def to_dict(self):
        return {
            'diff': self.diff,
            'lb': self.lb,
            'ub': self.ub,
            'from_init': self.from_init,
            'from_init_margin': self.from_init_margin,
            'stride': self.stride
        }


class PossibleNumericalError(Exception):
    OVERFLOW_LIMIT = 1e38
    UNDERFLOW_LIMIT = 1e-37
    OVERFLOW_D = 38
    UNDERFLOW_D = -37
    ERROR_UNKNOWN = -1
    ERROR_CONTAINS_ZERO = 0
    ERROR_OVERFLOW = 1
    ERROR_UNDERFLOW = 2
    OPs2Check = {"Exp", "Log", "Div", "Sqrt", "Pow", "Reciprocal", "Range", "NegativeLogLikelihoodLoss", "LogSoftmax"}
    code2str = {ERROR_CONTAINS_ZERO: "Range contains zero.", ERROR_OVERFLOW: "Operator overflows.",
                ERROR_UNDERFLOW: "Operator underflows.", ERROR_UNKNOWN: 'Unknown error.'}
    continue_prop = False

    def __init__(self, optype='', var_name='', cur_range='', err_cond=-1):
        self.optype = optype
        self.var_name = var_name
        self.cur_range = cur_range
        self.err_cond = err_cond
        self.message = f'Possible numerical error on op {self.optype} (variable: {self.var_name}): \n' \
                       f'  range from analysis is {self.cur_range};\n ' \
                       f'  triggered numerical error condition {self.code2str[self.err_cond]}'
        super(PossibleNumericalError, self).__init__(self.message)

    @staticmethod
    def is_invalid(x):
        return x.isnan().any() or x.isinf().any()

    @staticmethod
    def clip_to_valid_range(x, lb=None, ub=None):
        if not PossibleNumericalError.continue_prop:
            return None
        if lb is None:
            lb = -PossibleNumericalError.OVERFLOW_LIMIT
        if ub is None:
            ub = PossibleNumericalError.OVERFLOW_LIMIT
        old_lb = x.lb
        old_ub = x.ub
        valid_lb = torch.full(old_lb.shape, lb, device=old_lb.device, dtype=old_lb.dtype)
        valid_ub = torch.full(old_ub.shape, ub, device=old_ub.device, dtype=old_ub.dtype)
        x.lb = torch.where(old_lb.isnan() | old_ub.isnan(),
                           valid_lb,
                           torch.maximum(valid_lb, old_lb))
        x.ub = torch.where(old_lb.isnan() | old_ub.isnan(),
                           valid_ub,
                           torch.minimum(valid_ub, old_ub))
        del old_lb
        del old_ub
        if PossibleNumericalError.is_invalid(x.lb) or PossibleNumericalError.is_invalid(x.ub):
            return None
        return x


def get_numel(shape_list):
    ret = 1
    for s in shape_list: ret *= s
    return ret


def parse_attribute(node):
    """
        Parse the onnx node instance's attribute field to dict
    :param node:
    :return:
    """
    ans = dict()
    for item in node.attribute:
        ans[item.name] = get_attribute_value(item)
    return ans

discrete_types = ['UINT8', 'INT8', 'UINT16', 'INT16', 'INT32', 'INT64', 'STRING', 'BOOL']

unsupported_types = ['STRING']

datatype_mapping = dict([(id, x.name) for id, x in enumerate(onnx.TensorProto.DataType._enum_type.values)])

fine_grain_parameters = {
    # the parameters that need fine grain abstraction
    # in the format of k:op_type, v:index_of_inputs(0-base) that needs fine grain abstraction
    'Reshape': [1],
    'Slice': [1, 2, 3, 4],
    'Squeeze': [1],
    'Unsqueeze': [1],
    'Tile': [1, 2],
    'Loop': [0, 1],
    'SequenceInsert': [2],
    'ConstantOfShape': [0],
    'Gather': [1],
    'GatherND': [1],
    'ReduceSum': [1],
    'ScatterElements': [1],
    'Expand': [1],
    'Split': [1],
    'Pad': [1, 2],
    # for NLL loss, this item may be removed
    'NegativeLogLikelihoodLoss': [1],
    'Clip': [1, 2],
    'OneHot': [1],
    'Resize': [1, 2, 3],
}

# The exact number of following defined op_types can be easily derived from either fine abstraction or coarse abstraction
# Thus, we don't need to backflow the fine grain requirement through those op_types
forbid_fine_grain_flow = ['Shape', 'RandomNormalLike', 'RandomUniformLike']

# PyTorch exports some values like -INT_MAX or INT_MAX to represent -inf and +inf
# These values may become out of scope when converted to float than int64, and thus cause unprecedented errors
# To avoid this, when coverting float to int index, we first apply clipping with this threshold
# This threshold only needs to be large enough to cover all possible non-infty indexing
index_clip_thres = 99999999
