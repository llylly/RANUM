import unittest
import torch
import numpy as np

from interp.interp_utils import AbstractionInitConfig
from interp.interp_operator import Abstraction, Interpreter


def summary(obj: Abstraction):
    print('var_name:', obj.var_name)
    print('lb_tensor:', obj.lb)
    print('lb_tensor shape:', obj.lb.shape)
    print('ub_tensor:', obj.ub)
    print('ub_tensor shape:', obj.ub.shape)
    print('splits:', obj.splits)
    print('')


def tf_equal(a, b, EPS=1e-5):
    return np.linalg.norm((a.detach().numpy() - np.array(b)).reshape(-1)) < EPS


class TestAbstraction(unittest.TestCase):
    def abst_shape_check(self, obj: Abstraction):
        target_size = [len(x) for x in obj.splits]
        self.assertEqual(obj.get_dim(), len(obj.splits))
        self.assertEqual(obj.lb.dim(), obj.get_dim())
        self.assertEqual(obj.ub.dim(), obj.get_dim())
        self.assertEqual(obj.lb.shape, torch.Size(target_size))
        self.assertEqual(obj.ub.shape, torch.Size(target_size))

    def test_init(self):
        conf1 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0.1, stride=5)
        abst1 = Abstraction()
        abst1.load(conf1, 'v1', [10, 10], 'FLOAT', None)
        summary(abst1)

        conf2 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0.1, stride=5, from_init=True)
        abst2 = Abstraction()
        abst2.load(conf2, 'v2', [10, 10], 'FLOAT', np.array(range(100)).reshape(10, 10))
        summary(abst2)

        # checker
        self.assertTrue((abst1.lb.detach().numpy() == np.array([[-1., -1.], [-1., -1.]])).all())
        self.assertTrue((abst1.ub.detach().numpy() == np.array([[1., 1.], [1., 1.]])).all())

        self.assertTrue(tf_equal(abst2.lb, [[-0.1, 4.9], [49.9, 54.9]]))
        self.assertTrue(tf_equal(abst2.ub, [[44.1, 49.1], [94.1, 99.1]]))

        # assert np.linalg.norm((abst2.lb.detach().numpy() - np.array([[-0.1,4.9],[49.9,54.9]])).reshape(-1)) < 1e-5
        # assert np.linalg.norm((abst2.ub.detach().numpy() - np.array([[44.1,49.1],[94.1,99.1]])).reshape(-1)) < 1e-5

    def test_split(self):
        conf3 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0., stride=4, from_init=True)
        conf4 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0., stride=3, from_init=True)

        abst3 = Abstraction()
        abst3.load(conf3, 'v3', [12, 12], 'FLOAT', np.array(range(12 * 12)).reshape(12, 12))
        abst4 = Abstraction()
        abst4.load(conf4, 'v3', [12, 12], 'FLOAT', np.array(range(12 * 12)).reshape(12, 12))

        summary(abst3)

        abst3.split_by(abst4.splits)

        summary(abst3)
        local_EPS = 1e-6
        self.assertEqual(6, abst3.lb.shape[0])
        self.assertEqual(6, abst3.lb.shape[1])

        for i in range(abst3.lb.shape[0]):
            for j in range(abst3.lb.shape[1]):
                if i % 2 == 0 and j % 2 == 0:
                    self.assertLessEqual(abs(abst3.lb[i][j] - abst3.lb[i][j + 1]), local_EPS)
                    self.assertLessEqual(abs(abst3.lb[i][j] - abst3.lb[i + 1][j + 1]), local_EPS)
                    self.assertLessEqual(abs(abst3.lb[i][j] - abst3.lb[i + 1][j]), local_EPS)

                    self.assertLessEqual(abs(abst3.ub[i][j] - abst3.ub[i][j + 1]), local_EPS)
                    self.assertLessEqual(abs(abst3.ub[i][j] - abst3.ub[i + 1][j + 1]), local_EPS)
                    self.assertLessEqual(abs(abst3.ub[i][j] - abst3.ub[i + 1][j]), local_EPS)

    def test_requires_grad(self):
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        abst1 = Abstraction()
        abst1.load(conf1, 'v1', [2, 2, 3, 5], 'FLOAT', np.array(range(2 * 2 * 3 * 5)).reshape((2, 2, 3, 5)))
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=3)
        abst2 = Abstraction()
        abst2.load(conf2, 'v2', [2, 2, 3, 5], 'FLOAT', -np.array(range(2 * 2 * 3 * 5)).reshape((2, 2, 3, 5)))
        abst1.split_by(abst2.splits)
        self.assertTrue(abst1.lb.requires_grad and abst1.ub.requires_grad)
        abst1.smash()
        self.assertTrue(abst1.lb.requires_grad and abst1.ub.requires_grad)

    def test_MatMul(self):
        interp = Interpreter()

        stride = 2
        conf5 = AbstractionInitConfig(diff=True, from_init=True, stride=stride)
        abst5 = Abstraction()
        abst5.load(conf5, 'v5', [2, 2, 3, 5], 'FLOAT', np.array(range(2 * 2 * 3 * 5)).reshape((2, 2, 3, 5)))
        abst6 = Abstraction()
        abst6.load(conf5, 'v6', [1, 1, 5, 6], 'FLOAT', np.array(range(1 * 1 * 5 * 6)).reshape((1, 1, 5, 6)))

        abst7, _ = interp.interp_MatMul([abst5, abst6], None, 'MatMul', 'abst')

        self.abst_shape_check(abst7)
        self.assertEqual(abst7.lb.shape[2], (3 + 1) // stride)
        self.assertEqual(abst7.lb.shape[3], (6 + 1) // stride)
        summary(abst7)
        abst5.smash()
        abst6.smash()
        abst8, _ = interp.interp_MatMul([abst5, abst6], None, 'MatMul', 'abst_smash')
        summary(abst8)

        smashed_lb = abst8.lb[0][0][0][0].item()
        smashed_ub = abst8.ub[0][0][0][0].item()
        self.assertTrue((abst7.lb >= smashed_lb).all().item())
        self.assertTrue((abst7.ub <= smashed_ub).all().item())


if __name__ == '__main__':
    unittest.main()
