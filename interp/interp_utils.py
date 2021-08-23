
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
        # if from_init=True, the margin of abstraction from the init data, ususally set to 0
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

    def __init__(self, optype, varname, cur_range, err_cond):
        self.optype = optype
        self.varname = varname
        self.cur_range = cur_range
        self.err_cond = err_cond
        self.message = f'Possible numerical error on op {self.optype} (variable: {self.varname}): \n' \
                       f'  range from analysis is {self.cur_range};\n ' \
                       f'  triggered numerical error condition {self.err_cond}'
        super(PossibleNumericalError, self).__init__(self.message)


discrete_types = ['UINT8', 'INT8', 'UINT16', 'INT16', 'INT32', 'INT64', 'STRING', 'BOOL']

