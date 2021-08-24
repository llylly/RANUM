
import torch
import numpy as np

from interp.interp_utils import AbstractionInitConfig
from interp.interp_operator import Abstraction, Interpreter

def summary(obj):
    print('var_name:', obj.var_name)
    print('lb_tensor:', obj.lb)
    print('lb_tensor shape:', obj.lb.shape)
    print('ub_tensor:', obj.ub)
    print('ub_tensor shape:', obj.ub.shape)
    print('splits:', obj.splits)
    print('')

def abst_shape_check(obj: Abstraction):
    target_size = [len(x) for x in obj.splits]
    assert obj.get_dim() == len(obj.splits)
    assert obj.lb.dim() == obj.get_dim()
    assert obj.ub.dim() == obj.get_dim()
    assert obj.lb.shape == torch.Size(target_size)
    assert obj.ub.shape == torch.Size(target_size)

def tf_equal(a, b, EPS=1e-5):
    return np.linalg.norm((a.detach().numpy() - np.array(b)).reshape(-1)) < EPS

if __name__ == '__main__':

    interp = Interpreter()

    conf1 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0.1, stride=5)
    abst1 = Abstraction()
    abst1.load(conf1, 'v1', [10,10], 'FLOAT', None)
    summary(abst1)

    conf2 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0.1, stride=5, from_init=True)
    abst2 = Abstraction()
    abst2.load(conf2, 'v2', [10,10], 'FLOAT', np.array(range(100)).reshape(10,10))
    summary(abst2)

    # checker
    assert (abst1.lb.detach().numpy() == np.array([[-1.,-1.],[-1.,-1.]])).all()
    assert (abst1.ub.detach().numpy() == np.array([[1.,1.],[1.,1.]])).all()

    assert tf_equal(abst2.lb, [[-0.1,4.9],[49.9,54.9]])
    assert tf_equal(abst2.ub, [[44.1,49.1],[94.1,99.1]])

    # assert np.linalg.norm((abst2.lb.detach().numpy() - np.array([[-0.1,4.9],[49.9,54.9]])).reshape(-1)) < 1e-5
    # assert np.linalg.norm((abst2.ub.detach().numpy() - np.array([[44.1,49.1],[94.1,99.1]])).reshape(-1)) < 1e-5

    conf3 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0., stride=4, from_init=True)
    conf4 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0., stride=3, from_init=True)

    abst3 = Abstraction()
    abst3.load(conf3, 'v3', [12,12], 'FLOAT', np.array(range(12*12)).reshape(12,12))
    abst4 = Abstraction()
    abst4.load(conf4, 'v3', [12,12], 'FLOAT', np.array(range(12*12)).reshape(12,12))

    summary(abst3)

    abst3.split_by(abst4.splits)

    summary(abst3)

    for i in range(abst3.lb.shape[0]):
        for j in range(abst3.lb.shape[1]):
            if i % 2 == 0 and j % 2 == 0:
                assert abs(abst3.lb[i][j] - abst3.lb[i][j+1]) < 1e-6
                assert abs(abst3.lb[i][j] - abst3.lb[i+1][j+1]) < 1e-6
                assert abs(abst3.lb[i][j] - abst3.lb[i+1][j+1]) < 1e-6

                assert abs(abst3.ub[i][j] - abst3.ub[i][j+1]) < 1e-6
                assert abs(abst3.ub[i][j] - abst3.ub[i+1][j+1]) < 1e-6
                assert abs(abst3.ub[i][j] - abst3.ub[i+1][j+1]) < 1e-6

    conf5 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
    abst5 = Abstraction()
    abst5.load(conf5, 'v5', [2, 2, 3, 5], 'FLOAT', np.array(range(2*2*3*5)).reshape((2,2,3,5)))
    abst6 = Abstraction()
    abst6.load(conf5, 'v6', [1, 1, 5, 6], 'FLOAT', np.array(range(1*1*5*6)).reshape((1,1,5,6)))

    abst7, _ = interp.interp_MatMul([abst5, abst6], None, 'MatMul', 'abst')

    abst_shape_check(abst7)

    print('test passed')

