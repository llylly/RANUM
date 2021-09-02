import onnx
from onnx.helper import get_attribute_value

class AbstractionInitConfig(object):

    def __init__(self, diff: bool, lb=0., ub=1., from_init=False, from_init_margin=0., stride=-1):
        super(AbstractionInitConfig, self).__init__()
        # diff: whether requires_grad differentiable
        self.diff = diff
        # default lower bound of tensor values
        self.lb = lb
        # default upper bound of tensor values
        self.ub = ub
        # if real tensor values are present, whether to use them to determine lower and upper bounds
        self.from_init = from_init
        # if from_init=True, the margin of abstraction from the init data, usually set to 0
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
    CONTAINS_ZERO = 0
    code2str = {0: "Range contains zero."}
    def __init__(self, optype, var_name, cur_range, err_cond):
        self.optype = optype
        self.var_name = var_name
        self.cur_range = cur_range
        self.err_cond = err_cond
        self.message = f'Possible numerical error on op {self.optype} (variable: {self.var_name}): \n' \
                       f'  range from analysis is {self.cur_range};\n ' \
                       f'  triggered numerical error condition {self.code2str[self.err_cond]}'
        super(PossibleNumericalError, self).__init__(self.message)


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
    'Slice': [1,2,3,4],
    'Squeeze': [1],
    'Unsqueeze': [1]
}

# The exact number of following defined op_types can be easily derived from either fine abstraction or corase abstraction
# Thus, we don't need to backflow the fine grain requirement through those op_types
forbid_fine_grain_flow = ['Shape']


