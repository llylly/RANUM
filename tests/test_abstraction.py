import unittest
import torch
import numpy as np

from interp.interp_utils import AbstractionInitConfig
from interp.interp_operator import Abstraction, Interpreter


def summary(obj: Abstraction):
    obj.print()


def tf_equal(a, b, EPS=1e-5):
    # tensor float equal
    return np.linalg.norm((a.detach().numpy() - np.array(b)).reshape(-1)) < EPS


def correct_abstraction(abst: Abstraction, arr):
    """
        Return whether abst correctly abstracts the concrete tensor arr
    :param abst:
    :param arr:
    :return:
    """

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()

    lb = ub = arr
    # print(lb, ub)
    for i, item in enumerate(abst.splits):
        new_lb = list()
        new_ub = list()
        for j in range(len(item)):
            if j < len(item) - 1:
                now_item_l = np.take(lb, list(range(item[j], item[j+1])), axis=i)
                now_item_u = np.take(ub, list(range(item[j], item[j+1])), axis=i)
            else:
                now_item_l = np.take(lb, list(range(item[j], abst.shape[i])), axis=i)
                now_item_u = np.take(ub, list(range(item[j], abst.shape[i])), axis=i)
            new_lb.append(now_item_l.min(axis=i))
            new_ub.append(now_item_u.max(axis=i))
            # print(new_lb, new_ub)
        lb = np.stack(new_lb, axis=i)
        ub = np.stack(new_ub, axis=i)
        # print(lb.shape, ub.shape)

    # print(abst.splits)
    # print(lb)
    # print(abst.lb.detach().numpy())

    abst_lb = abst.lb.detach().numpy().reshape(-1)
    abst_ub = abst.ub.detach().numpy().reshape(-1)
    lb = lb.reshape(-1)
    ub = ub.reshape(-1)


    return all(abst_lb <= lb) and all(ub <= abst_ub)



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
        # summary(abst1)

        conf2 = AbstractionInitConfig(diff=True, lb=-1, ub=1, from_init_margin=0.1, stride=5, from_init=True)
        abst2 = Abstraction()
        abst2.load(conf2, 'v2', [10, 10], 'FLOAT', np.array(range(100)).reshape(10, 10))
        # summary(abst2)

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

        # summary(abst3)

        abst3.split_by(abst4.splits)

        # summary(abst3)
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

    def test_MatMul1(self):
        """
            Normal matrix multiplication
        :return:
        """
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
        # summary(abst7)
        abst5.smash()
        abst6.smash()
        abst8, _ = interp.interp_MatMul([abst5, abst6], None, 'MatMul', 'abst_smash')
        # summary(abst8)

        smashed_lb = abst8.lb[0][0][0][0].item()
        smashed_ub = abst8.ub[0][0][0][0].item()
        self.assertTrue((abst7.lb >= smashed_lb).all().item())
        self.assertTrue((abst7.ub <= smashed_ub).all().item())

    def test_MatMul2(self):
        """
            The boundary case: left dot-product and right dot-product
        :return:
        """
        interp = Interpreter()

        stride = 2
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=stride)
        abst1 = Abstraction()
        a = np.array(range(5))
        abst1.load(conf1, 'v1', [5], 'FLOAT', a)
        abst2 = Abstraction()
        b = np.array(range(5)) * 3.
        abst2.load(conf1, 'v2', [5], 'FLOAT', b)
        abst3 = Abstraction()
        c = np.array(range(25)).reshape((5,5)) + 1.
        abst3.load(conf1, 'v3', [5,5], 'FLOAT', c)

        out1, _ = interp.interp_MatMul([abst1, abst3], None, 'MatMul', 'out1')
        out2, _ = interp.interp_MatMul([abst3, abst2], None, 'Matmul', 'out2')

        gt1 = Abstraction()
        gt1.load(conf1, 'gt1', [5], 'FLOAT', a @ c)
        gt2 = Abstraction()
        gt2.load(conf1, 'gt2', [5], 'FLOAT', c @ b)

        self.assertTrue((out1.lb <= gt1.lb).all().item())
        self.assertTrue((gt1.ub <= out1.ub).all().item())
        self.assertTrue((out2.lb <= gt2.lb).all().item())
        self.assertTrue((gt2.ub <= out2.ub).all().item())

    def test_Reshape1(self):
        """
            Test general flatten
        :return:
        """
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        conf2 = AbstractionInitConfig(diff=False, from_init=True, stride=1)

        targ_shape = [5, 5, -1]

        a = np.array(list(range(5 * 5 * 5 * 5 * 2))).reshape((5, 5, 5, 5, 2))
        abst1 = Abstraction()
        abst1.load(conf1, 'v1', [5,5,5,5,2], 'FLOAT', a)

        b = a.reshape(tuple(targ_shape))
        abst2 = Abstraction()
        abst2.load(conf2, 'vshape', [3], 'INT', np.array(targ_shape))
        abst2, _ = interp.interp_Reshape([abst1, abst2], None, None, None)
        self.assertTrue(correct_abstraction(abst2, b))

        # =========

        targ_shape = [-1]
        c = a.reshape(tuple(targ_shape))
        abst3 = Abstraction()
        abst3.load(conf2, 'vshape', [1], 'INT', np.array(targ_shape))
        abst3, _ = interp.interp_Reshape([abst1, abst3], None, None, None)
        self.assertTrue(correct_abstraction(abst3, c))

        # summary(abst3)

        # ============

        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3,6])

        targ_shape = [-1]

        a = np.array(list(range(6 * 6))).reshape((6, 6))
        abst1 = Abstraction()
        abst1.load(conf3, 'v2', [6,6], 'FLOAT', a)

        d = a.reshape(tuple(targ_shape))
        abst4 = Abstraction()
        abst4.load(conf2, 'vshape', [1], 'INT', np.array(targ_shape))
        abst4, _ = interp.interp_Reshape([abst1, abst4], None, None, None)
        self.assertTrue(correct_abstraction(abst4, d))

        # summary(abst4)

    def test_Reshape2(self):
        """
            Test general stretch
        :return:
        """
        interp = Interpreter()
        conf_shape = AbstractionInitConfig(diff=False, from_init=True, stride=1)


        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=3)

        a = np.array(list(range(3 * 3 * 9))).reshape((3,3,9))
        abst1 = Abstraction()
        abst1.load(conf1, 'v1', [3,3,9], 'FLOAT', a)

        targ_shape = [3,3,3,3]
        b = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction()
        abst_shape.load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        abst2, _ = interp.interp_Reshape([abst1, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst2, b))

        # =============

        a = np.array(list(range(3 * 3 * 16))).reshape((3,3,16))
        abst3 = Abstraction().load(conf1, 'v2', [3,3,16], 'FLOAT', a)

        targ_shape = [3,3,4,4]
        c = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        abst4, _ = interp.interp_Reshape([abst3, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst4, c))

        # =============

        targ_shape = [3,3,2,2,4]
        d = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [5], 'INT', np.array(targ_shape))

        abst5, _ = interp.interp_Reshape([abst3, abst_shape], None, None, None)

        self.assertTrue(correct_abstraction(abst5, d))

        # =============

        targ_shape = [3,3,2,2,2,2]
        e = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [6], 'INT', np.array(targ_shape))

        abst6, _ = interp.interp_Reshape([abst3, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst6, e))

        # =============

        f = np.array(list(range(3 * 3 * 24))).reshape((3,3,24))
        abst7 = Abstraction().load(conf1, 'v3', [3,3,24], 'FLOAT', f)

        targ_shape = [3,3,4,6]
        g = f.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        abst8, _ = interp.interp_Reshape([abst7, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst8, g))

    def test_Reshape3(self):
        """
            Test irregular reshape
        :return:
        """
        interp = Interpreter()
        conf_shape = AbstractionInitConfig(diff=False, from_init=True, stride=1)

        a = np.array(list(range(5 * 5 * 12 * 12))).reshape((5,5,12,12))
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=3)
        abst1 = Abstraction().load(conf1, 'v1', [5,5,12,12], 'FLOAT', a)

        targ_shape = [5,5,3,48]
        b = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        abst2, _ = interp.interp_Reshape([abst1, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst2, b))

        # =============

        targ_shape = [5,5,6,24]
        c = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        targ_shape_1 = [5,5,144]
        d = a.reshape(tuple(targ_shape_1))
        abst_shape_1 = Abstraction().load(conf_shape, 'vshape_1', [3], 'INT', np.array(targ_shape_1))

        abst4, _ = interp.interp_Reshape([abst2, abst_shape_1], None, None, None)
        self.assertTrue(correct_abstraction(abst4, d))

        # =============

        abst3, _ = interp.interp_Reshape([abst2, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst3, c))

    def test_Reshape4(self):
        """
            Test reshape with forced resplit
        :return:
        """


        interp = Interpreter()
        interp.smash = 100

        conf_shape = AbstractionInitConfig(diff=False, from_init=True, stride=1)

        a = np.array(list(range(1600))).reshape((40, 40))
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=10)
        abst1 = Abstraction().load(conf1, 'v1', [40,40], 'FLOAT', a)

        targ_shape = [1,1600]
        abst_shape = Abstraction().load(conf_shape, 'vshape', [2], 'INT', np.array(targ_shape))

        abst2, _ = interp.interp_Reshape([abst1, abst_shape], None, None, None)
        summary(abst2)


if __name__ == '__main__':
    unittest.main()
