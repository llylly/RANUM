
import torch
import numpy as np

from interp.interp_utils import AbstractionInitConfig
from interp.interp_operator import Abstraction

def summary(obj):
    print('var_name:', obj.var_name)
    print('lb_tensor:', obj.lb)
    print('lb_tensor shape:', obj.lb.shape)
    print('ub_tensor:', obj.ub)
    print('ub_tensor shape:', obj.ub.shape)
    print('splits:', obj.splits)
    print('')
    # === DEBUG end ===

if __name__ == '__main__':

    conf1 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0.1, stride=5)
    abst1 = Abstraction(conf1, 'v1', [10,10], 'FLOAT', None)
    summary(abst1)

    conf1 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0.1, stride=5, from_init=True)
    abst2 = Abstraction(conf1, 'v2', [10,10], 'FLOAT', np.array(range(100)).reshape(10,10))
    summary(abst2)

    # checker
    assert (abst1.lb.detach().numpy() == np.array([[-1.,-1.],[-1.,-1.]])).all()
    assert (abst1.ub.detach().numpy() == np.array([[1.,1.],[1.,1.]])).all()

    assert np.linalg.norm((abst2.lb.detach().numpy() - np.array([[-0.1,4.9],[49.9,54.9]])).reshape(-1)) < 1e-5
    assert np.linalg.norm((abst2.ub.detach().numpy() - np.array([[44.1,49.1],[94.1,99.1]])).reshape(-1)) < 1e-5
