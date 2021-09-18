import unittest
import torch
import numpy as np
from onnx import helper, TensorProto
from functools import reduce, partial

from interp.interp_utils import AbstractionInitConfig, EPS, PossibleNumericalError
from interp.interp_operator import Abstraction, Interpreter


def summary(obj: Abstraction):
    obj.print()


def tf_equal(a, b):
    # tensor float equal
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    return np.linalg.norm((a - np.array(b)).reshape(-1), ord=1) < EPS


def correct_format(abst):
    if isinstance(abst.lb, list):
        ret = True
        for i in range(len(abst.lb)):
            indiv_abst = Abstraction()
            indiv_abst.lb = abst.lb[i]
            indiv_abst.ub = abst.ub[i]
            indiv_abst.shape = abst.shape[i]
            indiv_abst.splits = abst.splits[i]
            if not correct_format(indiv_abst):
                ret = False
                break
        return ret
    else:
        check_lb_1 = all([len(x) == y for x, y in zip(abst.splits, abst.lb.shape)])
        check_ub_1 = all([len(x) == y for x, y in zip(abst.splits, abst.ub.shape)])
        check_splits = all([all([z < y for z in x]) for x, y in zip(abst.splits, abst.shape)])
        return check_lb_1 and check_ub_1 and check_splits


def correct_abstraction(abst: Abstraction, arr, tight=False):
    """
        Return whether abst correctly abstracts the concrete tensor arr
    :param abst:
    :param arr:
    :return:
    """

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()

    if isinstance(arr, list):
        ans = True
        for i, item in enumerate(arr):
            now_abs = Abstraction()
            now_abs.lb = abst.lb[i]
            now_abs.ub = abst.ub[i]
            now_abs.splits = abst.splits[i]
            now_abs.shape = abst.shape[i]
            ans = ans and correct_abstraction(now_abs, item, tight)
        return ans

    lb = ub = arr
    # print(lb, ub)
    for i, item in enumerate(abst.splits):
        new_lb = list()
        new_ub = list()
        for j in range(len(item)):
            if j < len(item) - 1:
                now_item_l = np.take(lb, list(range(item[j], item[j + 1])), axis=i)
                now_item_u = np.take(ub, list(range(item[j], item[j + 1])), axis=i)
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

    if not tight:
        return all(abst_lb <= lb + EPS) and all(ub <= abst_ub + EPS)
    else:
        return tf_equal(abst_lb, lb) and tf_equal(abst_ub, ub)


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
        c = np.array(range(25)).reshape((5, 5)) + 1.
        abst3.load(conf1, 'v3', [5, 5], 'FLOAT', c)

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
        abst1.load(conf1, 'v1', [5, 5, 5, 5, 2], 'FLOAT', a)

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

        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3, 6])

        targ_shape = [-1]

        a = np.array(list(range(6 * 6))).reshape((6, 6))
        abst1 = Abstraction()
        abst1.load(conf3, 'v2', [6, 6], 'FLOAT', a)

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

        a = np.array(list(range(3 * 3 * 9))).reshape((3, 3, 9))
        abst1 = Abstraction()
        abst1.load(conf1, 'v1', [3, 3, 9], 'FLOAT', a)

        targ_shape = [3, 3, 3, 3]
        b = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction()
        abst_shape.load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        abst2, _ = interp.interp_Reshape([abst1, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst2, b))

        # =============

        a = np.array(list(range(3 * 3 * 16))).reshape((3, 3, 16))
        abst3 = Abstraction().load(conf1, 'v2', [3, 3, 16], 'FLOAT', a)

        targ_shape = [3, 3, 4, 4]
        c = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        abst4, _ = interp.interp_Reshape([abst3, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst4, c))

        # =============

        targ_shape = [3, 3, 2, 2, 4]
        d = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [5], 'INT', np.array(targ_shape))

        abst5, _ = interp.interp_Reshape([abst3, abst_shape], None, None, None)

        self.assertTrue(correct_abstraction(abst5, d))

        # =============

        targ_shape = [3, 3, 2, 2, 2, 2]
        e = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [6], 'INT', np.array(targ_shape))

        abst6, _ = interp.interp_Reshape([abst3, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst6, e))

        # =============

        f = np.array(list(range(3 * 3 * 24))).reshape((3, 3, 24))
        abst7 = Abstraction().load(conf1, 'v3', [3, 3, 24], 'FLOAT', f)

        targ_shape = [3, 3, 4, 6]
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

        a = np.array(list(range(5 * 5 * 12 * 12))).reshape((5, 5, 12, 12))
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=3)
        abst1 = Abstraction().load(conf1, 'v1', [5, 5, 12, 12], 'FLOAT', a)

        targ_shape = [5, 5, 3, 48]
        b = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        abst2, _ = interp.interp_Reshape([abst1, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst2, b))

        # =============

        targ_shape = [5, 5, 6, 24]
        c = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [4], 'INT', np.array(targ_shape))

        targ_shape_1 = [5, 5, 144]
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
        abst1 = Abstraction().load(conf1, 'v1', [40, 40], 'FLOAT', a)

        targ_shape = [1, 1600]
        b = a.reshape(tuple(targ_shape))
        abst_shape = Abstraction().load(conf_shape, 'vshape', [2], 'INT', np.array(targ_shape))

        abst2, _ = interp.interp_Reshape([abst1, abst_shape], None, None, None)
        self.assertTrue(correct_abstraction(abst2, b))

        # ===================

        c = np.array(list(range(35 * 35))).reshape((35, 35))
        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=5)
        abst3 = Abstraction().load(conf3, 'v2', [35, 35], 'FLOAT', c)
        # summary(abst3)

        abst3.force_resplit([list(range(0, 35, 7)), list(range(0, 35, 7))])
        self.assertTrue(correct_abstraction(abst3, c))

        abst3.force_resplit([list(range(0, 35, 2)), list(range(0, 35, 2))])
        self.assertTrue(correct_abstraction(abst3, c))

        # summary(abst3)

    def test_Shape(self):

        interp = Interpreter()

        a = np.zeros((5, 6, 7, 8))
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=-1)
        abst1 = Abstraction().load(conf1, 'v1', [5, 6, 7, 8], 'FLOAT', a)

        node = helper.make_node('Shape', ['v1'], ['s'], name='shape',
                                start=1)

        abst_shape, _ = interp.interp_Shape([abst1], node, 'Shape', 'shape')
        self.assertTrue(correct_abstraction(abst_shape, np.array([6, 7, 8])))

        node = helper.make_node('Shape', ['v2'], ['s'], name='shape',
                                end=-1)

        abst_shape, _ = interp.interp_Shape([abst1], node, 'Shape', 'shape')
        self.assertTrue(correct_abstraction(abst_shape, np.array([5, 6, 7])))

        node = helper.make_node('Shape', ['v2'], ['s'], name='shape',
                                start=1, end=-1)

        abst_shape, _ = interp.interp_Shape([abst1], node, 'Shape', 'shape')
        self.assertTrue(correct_abstraction(abst_shape, np.array([6, 7])))

    def test_Slice(self):
        interp = Interpreter()
        conf_def = AbstractionInitConfig(diff=True, from_init=True, stride=4)
        conf_precise = AbstractionInitConfig(diff=False, from_init=True, stride=1)

        x = np.random.randn(20, 10, 5).astype(np.float64)
        y = x[0:3, 0:10]

        abst_x = Abstraction().load(conf_def, 'v1', [20, 10, 5], 'FLOAT', x)
        # summary(abst_x)
        node = helper.make_node(
            'Slice', ['v1'], ['s'], 'slice', starts=[0, 0], ends=[3, 10], steps=[1, 1]
        )
        abst_slice, _ = interp.interp_Slice([abst_x], node, 'Slice', 'slice')
        self.assertTrue(correct_abstraction(abst_slice, y))

        abst_starts = Abstraction().load(conf_precise, 'starts', [2], 'INT', np.array([0, 0]))
        abst_ends = Abstraction().load(conf_precise, 'ends', [2], 'INT', np.array([3, 10]))
        abst_axes = Abstraction().load(conf_precise, 'axes', [2], 'INT', np.array([0, 1]))
        abst_steps = Abstraction().load(conf_precise, 'steps', [2], 'INT', np.array([1, 1]))
        new_node = helper.make_node(
            'Slice', ['v1'], ['s'], 'slice'
        )
        new_abst_slice, _ = interp.interp_Slice([abst_x, abst_starts, abst_ends, abst_axes, abst_steps], new_node,
                                                'Slice', 'new_slice')
        self.assertTrue(correct_abstraction(new_abst_slice, y))

        starts = [10, 2]
        ends = [1000, 1000]
        steps = [2, 3]
        axis = [0, 1]
        node = helper.make_node(
            'Slice', ['v2'], ['s'], 'slice', starts=starts, ends=ends, steps=steps
        )
        abst_slice, _ = interp.interp_Slice([abst_x], node, 'Slice', 'slice')
        # summary(abst_slice)
        self.assertTrue(correct_abstraction(abst_slice, x[10:1000:2, 2:1000:3]))

        starts = [20, 20, 20]
        ends = [-10000, -10000, -10000]
        steps = [-1, -2, -1]
        node = helper.make_node(
            'Slice', ['v2'], ['s'], 'slice', starts=starts, ends=ends, steps=steps
        )
        abst_slice, _ = interp.interp_Slice([abst_x], node, 'Slice', 'slice')
        # summary(abst_slice)
        self.assertTrue(correct_abstraction(abst_slice, x[20:-10000:-1, 20:-10000:-2, 20:-10000:-1]))

        starts = [20, 20]
        ends = [-1000, -1000]
        steps = [-3, -2]
        axes = [1, 2]
        node = helper.make_node(
            'Slice', ['v2'], ['s'], 'slice', starts=starts, ends=ends, steps=steps, axes=axes
        )
        abst_slice, _ = interp.interp_Slice([abst_x], node, 'Slice', 'slice')
        # summary(abst_slice)
        self.assertTrue(correct_abstraction(abst_slice, x[:, 20:-1000:-3, 20:-1000:-2]))

    def test_Squeeze(self):
        interp = Interpreter()
        conf_def = AbstractionInitConfig(diff=True, from_init=True, stride=4)
        conf_precise = AbstractionInitConfig(diff=False, from_init=True, stride=1)

        x = np.random.randn(20, 10, 1, 5, 3, 1, 2).astype(np.float64)
        axes = [2, -2]
        node = helper.make_node(
            'Squeeze', ['v1'], ['s'], 'squeeze'
        )

        abst_x = Abstraction().load(conf_def, 'v1', [20, 10, 1, 5, 3, 1, 2], 'FLOAT', x)
        abst_axes = Abstraction().load(conf_precise, 'vaxes', [2], 'INT', np.array([2, 5]))

        abst_y, _ = interp.interp_Squeeze([abst_x, abst_axes], node, 'Squeeze', 'y')

        node = helper.make_node(
            'Squeeze', ['v1'], ['s'], 'squeeze', axes=axes
        )
        abst_y_new, _ = interp.interp_Squeeze([abst_x], node, 'Squeeze', 'y')

        node = helper.make_node(
            'Squeeze', ['v1'], ['s'], 'squeeze'
        )
        abst_y_new_new, _ = interp.interp_Squeeze([abst_x], node, 'Squeeze', 'y')

        self.assertTrue(correct_abstraction(abst_y, x.squeeze(axes[0]).squeeze(axes[1])))
        self.assertTrue(correct_abstraction(abst_y_new, x.squeeze(axes[0]).squeeze(axes[1])))
        self.assertTrue(correct_abstraction(abst_y_new_new, x.squeeze(axes[0]).squeeze(axes[1])))

    def test_Unsqueeze(self):
        interp = Interpreter()
        conf_def = AbstractionInitConfig(diff=True, from_init=True, stride=4)
        conf_precise = AbstractionInitConfig(diff=False, from_init=True, stride=1)

        x = np.random.randn(20, 10, 5).astype(np.float64)
        axes = [2, -2]
        node = helper.make_node(
            'Unsqueeze', ['v1'], ['s'], 'unsqueeze'
        )

        abst_x = Abstraction().load(conf_def, 'v1', [20, 10, 5], 'FLOAT', x)
        abst_axes = Abstraction().load(conf_precise, 'vaxes', [2], 'INT', np.array([2, -2]))

        abst_y, _ = interp.interp_Unsqueeze([abst_x, abst_axes], node, 'Unsqueeze', 'y')

        self.assertTrue(correct_abstraction(abst_y, x.reshape((20, 1, 10, 1, 5))))

        node = helper.make_node(
            'Unsqueeze', ['v1'], ['s'], 'unsqueeze', axes=axes
        )

        abst_y_new, _ = interp.interp_Unsqueeze([abst_x, abst_axes], node, 'Unsqueeze', 'y')

        self.assertTrue(correct_abstraction(abst_y_new, x.reshape((20, 1, 10, 1, 5))))

    def test_Concat(self):
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=30)
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=40)

        x = np.random.randn(260, 5, 260).astype(np.float64)
        y = np.random.randn(260, 3, 260).astype(np.float64)

        abst_x = Abstraction().load(conf1, 'x', [260, 5, 260], 'FLOAT', x)
        abst_y = Abstraction().load(conf2, 'y', [260, 3, 260], 'FLOAT', y)

        node = helper.make_node(
            'Concat', ['v1'], ['s'], 'concat', axis=1
        )

        abst_z, _ = interp.interp_Concat([abst_x, abst_y], node, 'Concat', 'z')
        self.assertTrue(correct_abstraction(abst_z, np.concatenate([x, y], axis=1)))

    def test_Reciprocal(self):
        interp = Interpreter()
        # test without errors
        x = np.abs(np.random.randn(10, 20, 30)).astype(np.float64) + 1
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=5)
        abst_x = Abstraction().load(conf1, 'x', [10, 20, 30], 'FLOAT', x)
        node = helper.make_node(
            'Reciprocal', ['x'], ['a'], 'Reciprocal0'
        )
        abst_z, exceptions = interp.interp_Reciprocal([abst_x], node, 'Reciprocal', 'y')
        self.assertTrue(correct_abstraction(abst_z, 1 / x, True))
        self.assertEqual(0, len(exceptions))

        # test with an error
        x[np.random.randint(10), np.random.randint(20), np.random.randint(30)] = -1
        abst_x2 = Abstraction().load(conf1, 'x', [10, 20, 30], 'FLOAT', x)
        abst_z, exceptions = interp.interp_Reciprocal([abst_x2], node, 'Reciprocal', 'z')
        self.assertIsNone(abst_z)
        self.assertEqual(1, len(exceptions))
        self.assertEqual(PossibleNumericalError.ERROR_CONTAINS_ZERO, exceptions[0].err_cond)

    def test_TanhAbsReluExp(self):
        interp = Interpreter()
        x = np.random.randn(10, 20, 30).astype(np.float64)
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=5)
        abst_x = Abstraction().load(conf1, 'x', [10, 20, 30], 'FLOAT', x)
        ops = [lambda x: np.tanh(x), lambda x: np.abs(x), lambda x: np.maximum(x, 0), lambda x: np.exp(x),
               lambda x: np.where(x >= 0,
                                  1 / (1 + np.exp(-x)),
                                  np.exp(x) / (1 + np.exp(x))),
               lambda x: -x]
        op_names = ["Tanh", "Abs", "Relu", "Exp", "Sigmoid", "Neg"]
        op_interps = [interp.interp_Tanh, interp.interp_Abs, interp.interp_Relu, interp.interp_Exp,
                      interp.interp_Sigmoid, interp.interp_Neg]
        tights = [True, False, True, True, True, True]

        for op, op_name, op_interp, tight in zip(ops, op_names, op_interps, tights):
            node = helper.make_node(
                op_name, ['x'], ['a'], op_name + "0"
            )
            z = op(x)
            abst_z, _ = op_interp([abst_x], node, op_name, 'z')
            self.assertTrue(correct_abstraction(abst_z, z, tight))

            if op_name == 'Abs':
                # test tightness
                z = np.abs(x) + 0.1
                abst_z = Abstraction().load(conf1, 'z', [10, 20, 30], 'FLOAT', z)
                abst_zz, _ = op_interp([abst_z], node, op_name, 'zz')
                self.assertTrue(correct_abstraction(abst_zz, z, True))

                z = -np.abs(x) - 0.1
                abst_z = Abstraction().load(conf1, 'z', [10, 20, 30], 'FLOAT', z)
                abst_zz, _ = op_interp([abst_z], node, op_name, 'zz')
                self.assertTrue(correct_abstraction(abst_zz, -z, True))

            if op_name == "Exp":  # test exp with error
                y = np.random.randn(10, 20, 30).astype(np.float32)
                y[0, 0, 0] = 100
                abst_y = Abstraction().load(conf1, 'y', [10, 20, 30], 'FLOAT', y)
                abst_z, exceptions = op_interp([abst_y], node, op_name, 'z')
                self.assertIsNone(abst_z)
                self.assertEqual(1, len(exceptions))
                self.assertEqual(PossibleNumericalError.ERROR_OVERFLOW, exceptions[0].err_cond)

    def test_AddSubMulDiv(self):
        interp = Interpreter()
        x = np.random.randn(10, 20, 10, 30).astype(np.float64)
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=5)
        abst_x = Abstraction().load(conf1, 'x', [10, 20, 10, 30], 'FLOAT', x)
        y = np.random.randn(1, 10, 1).astype(np.float64)
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        ops = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y]
        op_names = ["Add", "Sub", "Mul", "Div"]
        op_interps = [interp.interp_Add, interp.interp_Sub, interp.interp_Mul, interp.interp_Div]
        for op, op_name, op_interp in zip(ops, op_names, op_interps):
            node = helper.make_node(
                op_name, ['x', 'y'], ['a'], op_name + "0"
            )
            if op_name == "Div":  # Test div without errors
                abst_y = Abstraction().load(conf2, 'y', [1, 10, 1], 'FLOAT', abs(y) + 1)
                z = op(x, abs(y) + 1)
            else:
                abst_y = Abstraction().load(conf2, 'y', [1, 10, 1], 'FLOAT', y)
                z = op(x, y)
            abst_z, _ = op_interp([abst_x, abst_y], node, op_name, 'z')
            self.assertTrue(correct_abstraction(abst_z, z, False))

            # test div with an error
            if op_name == "Div":
                y = abs(y) + 1
                y[0, np.random.randint(10), 0] = -1
                abst_y = Abstraction().load(conf2, 'y', [1, 10, 1], 'FLOAT', y)
                abst_z, exceptions = op_interp([abst_x, abst_y], node, op_name, 'z')
                self.assertIsNone(abst_z)
                self.assertEqual(1, len(exceptions))
                self.assertEqual(PossibleNumericalError.ERROR_CONTAINS_ZERO, exceptions[0].err_cond)

    def test_Pow(self):
        interp = Interpreter()
        x = np.abs(np.random.randn(10, 20, 10, 30).astype(np.float64)) + 0.1
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=5)
        abst_x = Abstraction().load(conf1, 'x', [10, 20, 10, 30], 'FLOAT', x)
        y = (np.random.randn(1, 10, 1) * 10).astype(np.int32)
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        op_name = "Pow"
        node = helper.make_node(
            op_name, ['x', 'y'], ['a'], op_name + "0"
        )
        abst_y = Abstraction().load(conf2, 'y', [1, 10, 1], 'INT', y)
        # print(abst_y.lb, abst_y.ub)
        z = np.power(x, y)
        abst_z, _ = interp.interp_Pow([abst_x, abst_y], node, op_name, 'z')
        self.assertTrue(correct_abstraction(abst_z, z, False))

        # test Pow with an error
        y[0, 0, 0] = -1
        x[0, 0, 0, 0] = -1
        abst_x = Abstraction().load(conf1, 'x', [10, 20, 10, 30], 'FLOAT', x)
        abst_y = Abstraction().load(conf2, 'y', [1, 10, 1], 'INT', y)
        abst_z, exceptions = interp.interp_Pow([abst_x, abst_y], node, op_name, 'z')
        self.assertIsNone(abst_z)
        self.assertEqual(1, len(exceptions))
        self.assertEqual(PossibleNumericalError.ERROR_CONTAINS_ZERO, exceptions[0].err_cond)

        # test Pow with an error
        y[0, 0, 0] = 100
        x[0, 0, 0, 0] = 3
        abst_x = Abstraction().load(conf1, 'x', [10, 20, 10, 30], 'FLOAT', x)
        abst_y = Abstraction().load(conf2, 'y', [1, 10, 1], 'INT', y)
        abst_z, exceptions = interp.interp_Pow([abst_x, abst_y], node, op_name, 'z')
        self.assertIsNone(abst_z)
        self.assertEqual(1, len(exceptions))
        self.assertEqual(PossibleNumericalError.ERROR_OVERFLOW, exceptions[0].err_cond)

    def test_ConstantOfShape(self):
        # test case 1

        interp = Interpreter()
        x = np.random.randint(1, 20, 3).astype(np.int32)
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        abst_x = Abstraction().load(conf1, 'x', [3], 'INT', x)
        node = helper.make_node(
            'ConstantOfShape', ['x'], ['a'], 'ConstantOfShape0', value=123
        )
        abst_z, _ = interp.interp_ConstantOfShape([abst_x], node, 'ConstantOfShape', 'y')
        self.assertEqual(123, abst_z.lb[0][0][0].item())
        self.assertEqual(123, abst_z.ub[0][0][0].item())
        self.assertEqual(list(x), abst_z.shape)

        # test case 2

        x = np.array([4, 3, 2]).astype(np.int64)
        tensor_value = helper.make_tensor("value", TensorProto.FLOAT,
                                          [1], [123])
        node = helper.make_node(
            'ConstantOfShape',
            inputs=['x'],
            outputs=['y'],
            value=tensor_value,
        )

        y = np.ones(x, dtype=np.float64)
        abst_x = Abstraction().load(conf1, 'x', [3], 'INT', x)
        abst_zz, _ = interp.interp_ConstantOfShape([abst_x], node, 'ConstantOfShape', 'zz')
        self.assertEqual(123, abst_zz.lb[0][0][0].item())
        self.assertEqual(123, abst_zz.ub[0][0][0].item())
        self.assertEqual(list(x), abst_zz.shape)

    def test_RandomUniformLike(self):
        interp = Interpreter()
        x = np.random.randn(10, 20, 10, 30).astype(np.float64)
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=5)
        abst_x = Abstraction().load(conf1, 'x', [10, 20, 10, 30], 'FLOAT', x)
        node = helper.make_node(
            'RandomUniformLike', ['x'], ['a'], 'RandomUniformLike', low=-100, high=123,
        )
        abst_z, _ = interp.interp_RandomUniformLike([abst_x], node, 'RandomUniformLike', 'y')
        self.assertEqual(-100, abst_z.lb[0][0][0][0].item())
        self.assertEqual(123, abst_z.ub[0][0][0][0].item())
        self.assertEqual(list(x.shape), abst_z.shape)
        self.assertTrue(correct_format(abst_z))

    def test_BoolOps(self):
        for stride in [1, 2]:
            interp = Interpreter()
            x = np.random.randn(10, 20, 10, 30).astype(np.float64)
            conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=stride)
            abst_x = Abstraction().load(conf1, 'x', [10, 20, 10, 30], 'FLOAT', x)
            y = np.random.randn(1, 10, 1).astype(np.float64)
            conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=stride)
            abst_y = Abstraction().load(conf2, 'y', [1, 10, 1], 'FLOAT', y)
            ops = [lambda x, y: x < y, lambda x, y: x <= y, lambda x, y: x > y, lambda x, y: x >= y]
            op_names = ["Less", "LessOrEqual", "Greater", "GreaterOrEqual"]
            op_interps = [interp.interp_Less, interp.interp_LessOrEqual, interp.interp_Greater,
                          interp.interp_GreaterOrEqual]

            not_node = helper.make_node(
                "Not", ['x'], ['a'], "Not:0"
            )
            for op, op_name, op_interp in zip(ops, op_names, op_interps):
                node = helper.make_node(
                    op_name, ['x', 'y'], ['a'], op_name + "0"
                )
                z = op(x, y)
                abst_z, _ = op_interp([abst_x, abst_y], node, op_name, 'z')
                self.assertTrue(correct_abstraction(abst_z, z, stride == 1))

                z = ~z
                abst_z, _ = interp.interp_Not([abst_z], not_node, "Not", 'not_z')
                self.assertTrue(correct_abstraction(abst_z, z, stride == 1))

    def test_MinMax(self):
        interp = Interpreter()
        ops = [lambda x, y: np.minimum(x, y), lambda x, y: np.maximum(x, y)]
        op_names = ["Min", "Max"]
        op_interps = [interp.interp_Min, interp.interp_Max]
        for arg_nums in [1, 2, 3]:
            xs = [np.random.randn(10 if np.random.rand() < 0.5 else 1,
                                  12 if np.random.rand() < 0.5 else 1,
                                  13 if np.random.rand() < 0.5 else 1) if np.random.rand() < 0.5 else
                  np.random.randn(12 if np.random.rand() < 0.5 else 1,
                                  13 if np.random.rand() < 0.5 else 1)
                  for _ in range(arg_nums)]
            abst_xs = [
                Abstraction().load(AbstractionInitConfig(diff=True,
                                                         from_init=True,
                                                         stride=i + 1),
                                   'x', x.shape, 'FLOAT', x)
                for i, x in enumerate(xs)]
            for op, op_name, op_interp in zip(ops, op_names, op_interps):
                node = helper.make_node(
                    op_name, ['x%d' % i for i in range(arg_nums)], ['a'], op_name + "0"
                )
                z = reduce(op, xs[1:], xs[0])
                abst_z, _ = op_interp(abst_xs, node, op_name, 'z')
                self.assertTrue(correct_abstraction(abst_z, z))

    def test_ReduceMinMaxSum(self):
        for axes in [[0, -1], [1]]:
            for keepdims in [0, 1]:
                for stride in [1, 2]:
                    interp = Interpreter()
                    ops = [lambda x: partial(np.min, x), lambda x: partial(np.max, x), lambda x: partial(np.sum, x),
                           lambda x: partial(np.mean, x)]
                    op_names = ["ReduceMin", "ReduceMax", "ReduceSum", "ReduceMean"]
                    op_interps = [interp.interp_ReduceMin, interp.interp_ReduceMax, interp.interp_ReduceSum,
                                  interp.interp_ReduceMean]
                    x = np.random.randn(10, 11, 12, 13).astype(np.float64)
                    conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=stride)
                    abst_x = Abstraction().load(conf1, 'x', [10, 11, 12, 13], 'FLOAT', x)

                    for op, op_name, op_interp in zip(ops, op_names, op_interps):
                        if op_name != 'ReduceSum':
                            node = helper.make_node(
                                op_name, ['x'], ['a'], op_name + "0", axes=axes, keepdims=keepdims
                            )
                        else:
                            node = helper.make_node(
                                op_name, ['x'], ['a'], op_name + "0", keepdims=keepdims
                            )
                        abst_axes = Abstraction().load(AbstractionInitConfig(diff=False, from_init=True, stride=1),
                                                       'axes',
                                                       np.array(axes).shape, 'FLOAT', np.array(axes))
                        axes = [(axis + 4) % 4 for axis in axes]
                        axes.sort()
                        z = x
                        for axis in axes[::-1]:
                            z = op(z)(keepdims=keepdims == 1, axis=axis)
                        if op_name != 'ReduceSum':
                            abst_z, _ = op_interp([abst_x], node, op_name, 'z')
                        else:
                            abst_z, _ = op_interp([abst_x, abst_axes], node, op_name, 'z')
                        self.assertTrue(correct_abstraction(abst_z, z, stride == 1))

    def test_Softmax(self):
        for axis in [0, -1, 1]:
            for stride in [1, 2]:
                interp = Interpreter()
                x = np.random.randn(10, 11, 12).astype(np.float32)
                conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=stride)
                abst_x = Abstraction().load(conf1, 'x', [10, 11, 12], 'FLOAT', x)

                node = helper.make_node(
                    "Softmax", ['x'], ['a'], "Softmax0", axis=axis
                )
                z = torch.softmax(torch.Tensor(x), dim=axis).numpy()
                abst_z, _ = interp.interp_Softmax([abst_x], node, "Softmax", 'z')
                self.assertTrue(correct_abstraction(abst_z, z, stride == 1))

    def test_Tile(self):
        interp = Interpreter()
        x = np.random.randn(10, 12, 13)
        steps = [2, 3, 3]
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=5)
        conf_precise = AbstractionInitConfig(diff=False, from_init=True, stride=1)
        abst_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        abst_steps = Abstraction().load(conf_precise, 'steps', np.array(steps).shape, 'INT', np.array(steps))
        abst_y, _ = interp.interp_Tile([abst_x, abst_steps], None, 'Tile', 'y')
        self.assertTrue(correct_abstraction(abst_y, np.tile(x, steps)))

        steps = [3, 2]
        axes = [-1, -3]
        abst_steps = Abstraction().load(conf_precise, 'steps', np.array(steps).shape, 'INT', np.array(steps))
        abst_axes = Abstraction().load(conf_precise, 'axes', np.array(axes).shape, 'INT', np.array(axes))
        abst_z, _ = interp.interp_Tile([abst_x, abst_steps, abst_axes], None, 'Tile', 'z')
        self.assertTrue(correct_abstraction(abst_z, np.tile(x, [2, 1, 3])))

    def test_floor_ceil(self):
        interp = Interpreter()
        x = np.random.randn(8, 9, 10)
        conf = AbstractionInitConfig(diff=True, from_init=True, stride=3)

        # precise mode (default mode) has no grad
        abst_x = Abstraction().load(conf, 'x', x.shape, 'FLOAT', x)
        abst_floor, _ = interp.interp_Floor([abst_x], None, None, 'floor')
        self.assertTrue(correct_abstraction(abst_floor, np.floor(x), tight=True))
        abst_ceil, _ = interp.interp_Ceil([abst_x], None, None, 'ceil')
        self.assertTrue(correct_abstraction(abst_ceil, np.ceil(x), tight=True))

        # coarse and identical mode has grad
        # but coarse is imprecise, and identical loses soundness
        interp = Interpreter(ceil='coarse', floor='coarse')
        abst_floor, _ = interp.interp_Floor([abst_x], None, None, 'floor')
        self.assertTrue(correct_abstraction(abst_floor, np.floor(x)))
        abst_ceil, _ = interp.interp_Ceil([abst_x], None, None, 'ceil')
        self.assertTrue(correct_abstraction(abst_ceil, np.ceil(x)))

        torch.sum(abst_floor.lb).backward()
        self.assertTrue(tf_equal(abst_x.lb.grad, torch.ones_like(abst_x.lb.grad)))
        torch.sum(abst_ceil.ub).backward()
        self.assertTrue(tf_equal(abst_x.ub.grad, torch.ones_like(abst_x.ub.grad)))

    def test_sequence_insert(self):
        interp = Interpreter()
        x1 = np.random.randn(2, 2, 2)
        x2 = np.random.randn(3, 3, 3)
        x3 = np.random.randn(4, 4, 4)
        x4 = np.random.randn(5, 5, 5)
        conf = AbstractionInitConfig(diff=True, from_init=True, stride=2)

        abst_xseq = Abstraction().load(conf, 'xseq', x1.shape, 'FLOAT', [x1, x2, x3])

        # (abst_xseq).print()

        abst_x4 = Abstraction().load(conf, 'x4', x4.shape, 'FLOAT', x4)

        abst_ans, _ = interp.interp_SequenceInsert([abst_xseq, abst_x4], None, 'SequenceInsert', 'ans')
        self.assertListEqual(abst_ans.shape, [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])

        index = np.array(2.)
        abst_index = Abstraction().load(conf, 'index', index.shape, 'INT', index)
        abst_ans2, _ = interp.interp_SequenceInsert([abst_xseq, abst_x4, abst_index], None, 'SequenceInsert', 'ans')
        self.assertListEqual(abst_ans2.shape, [[2, 2, 2], [3, 3, 3], [5, 5, 5], [4, 4, 4]])

    def test_device_inheritance(self):
        # Todo
        interp = Interpreter()
        for cuda in [False, True]:
            for op in dir(interp):
                if op.startswith("interp"):
                    pass

    def test_sum(self):
        interp = Interpreter()

        x1 = np.random.randn(15, 15, 15)
        x2 = np.random.randn(15, 15, 15)
        x3 = np.random.randn(15, 15, 15)
        ans = x1 + x2 + x3

        for s1, s2, s3 in [(1, 1, 1), (1, 2, 3), (2, 3, 5), (3, 2, 4), (-1, -1, -1)]:
            conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=s1)
            conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=s2)
            conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=s3)

            abst_x1 = Abstraction().load(conf1, 'x1', x1.shape, 'FLOAT', x1)
            abst_x2 = Abstraction().load(conf2, 'x2', x2.shape, 'FLOAT', x2)
            abst_x3 = Abstraction().load(conf3, 'x3', x3.shape, 'FLOAT', x3)

            abst_ans, _ = interp.interp_Sum([abst_x1, abst_x2, abst_x3], None, 'Sum', 'ans')

            self.assertTrue(correct_format(abst_ans))
            self.assertTrue(correct_abstraction(abst_ans, ans, s1 == 1 and s2 == 1 and s3 == 1))

    def test_log(self):
        interp = Interpreter()

        x1 = np.random.randn(10, 10, 10)
        y1 = np.exp(x1)

        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        abst_y1 = Abstraction().load(conf1, 'y1', y1.shape, 'FLOAT', y1)
        abst_ans1, _ = interp.interp_Log([abst_y1], None, 'Lop', 'ans1')

        self.assertTrue(correct_abstraction(abst_ans1, x1, True))

        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        abst_y2 = Abstraction().load(conf2, 'y2', y1.shape, 'FLOAT', y1)
        abst_ans2, _ = interp.interp_Log([abst_y2], None, 'Lop', 'ans1')

        self.assertTrue(correct_abstraction(abst_ans2, x1))

        abst_x1 = Abstraction().load(conf2, 'x1', x1.shape, 'FLOAT', x1)
        abst_ans3, err = interp.interp_Log([abst_x1], None, 'Lop', 'ans3')
        self.assertEqual(len(err), 1)

    def test_transpose(self):
        interp = Interpreter()
        x = np.random.randn(5, 12, 9)
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        abst_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        node = helper.make_node('Transpose', ['x'], ['y'], name='transpose')
        y = np.transpose(x)
        abst_y, _ = interp.interp_Transpose([abst_x], node, 'Transpose', 'y')
        self.assertTrue(correct_abstraction(abst_y, y))
        self.assertTrue(correct_format(abst_y))
        self.assertListEqual(abst_y.shape, [9, 12, 5])

        # =========

        node = helper.make_node('Transpose', ['x'], ['y'], name='transpose', perm=[2, 0, 1])
        y = np.transpose(x, [2, 0, 1])
        abst_y, _ = interp.interp_Transpose([abst_x], node, 'Transpose', 'y')
        self.assertTrue(correct_abstraction(abst_y, y))
        self.assertTrue(correct_format(abst_y))
        self.assertListEqual(abst_y.shape, [9, 5, 12])

    def test_gather(self):
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=3)
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=4)
        conf_precise = AbstractionInitConfig(diff=True, from_init=True, stride=1)

        x = np.random.randn(10, 10, 10)
        abst_x1 = Abstraction().load(conf1, 'x1', x.shape, 'FLOAT', x)
        abst_x2 = Abstraction().load(conf2, 'x2', x.shape, 'FLOAT', x)

        ind = np.array([[1, 2, 1], [0, 1, 2], [3, 4, 9], [4, 5, 9]])
        abst_ind = Abstraction().load(conf_precise, 'gather', ind.shape, 'FLOAT', ind)

        node = helper.make_node(
            "Gather", ['x', 'index'], ['res'], "Gather", axis=1
        )

        abst_res, _ = interp.interp_Gather([abst_x1, abst_ind], node, 'Gather', 'res')
        res = x.take(ind.reshape(-1), axis=1).reshape((10,) + ind.shape + (10,))

        # abst_res.print()
        self.assertTrue(correct_abstraction(abst_res, res))
        self.assertTrue(correct_format(abst_res))
        self.assertListEqual(abst_res.splits[1], [0, 2])
        self.assertListEqual(abst_res.splits[2], [0, 2])

        # ===========

        node = helper.make_node(
            "Gather", ['x', 'index'], ['res'], "Gather", axis=0
        )

        abst_res, _ = interp.interp_Gather([abst_x1, abst_ind], node, 'Gather', 'res')
        res = x.take(ind.reshape(-1), axis=0).reshape(ind.shape + (10, 10))

        # abst_res.print()
        self.assertTrue(correct_abstraction(abst_res, res))
        self.assertTrue(correct_format(abst_res))
        self.assertListEqual(abst_res.splits[0], [0, 2])
        self.assertListEqual(abst_res.splits[1], [0, 2])

        # ===========

        ind = np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7]]])
        abst_ind = Abstraction().load(conf_precise, 'gather', ind.shape, 'FLOAT', ind)

        node = helper.make_node(
            "Gather", ['x', 'index'], ['res'], "Gather", axis=1
        )

        abst_res, _ = interp.interp_Gather([abst_x2, abst_ind], node, 'Gather', 'res')
        res = x.take(ind.reshape(-1), axis=1).reshape((10,) + ind.shape + (10,))

        # abst_res.print()
        self.assertTrue(correct_abstraction(abst_res, res))
        self.assertTrue(correct_format(abst_res))
        self.assertListEqual(abst_res.splits[1], [0])
        self.assertListEqual(abst_res.splits[2], [0, 2])
        self.assertListEqual(abst_res.splits[3], [0])

        # ===========

        ind = np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [3, 0, 1, 2], [3, 0, 1, 2]]])
        abst_ind = Abstraction().load(conf_precise, 'gather', ind.shape, 'FLOAT', ind)

        node = helper.make_node(
            "Gather", ['x', 'index'], ['res'], "Gather", axis=1
        )

        abst_res, _ = interp.interp_Gather([abst_x2, abst_ind], node, 'Gather', 'res')
        res = x.take(ind.reshape(-1), axis=1).reshape((10,) + ind.shape + (10,))

        # abst_res.print()
        self.assertTrue(correct_abstraction(abst_res, res))
        self.assertTrue(correct_format(abst_res))
        self.assertListEqual(abst_res.splits[1], [0])
        self.assertListEqual(abst_res.splits[2], [0])
        self.assertListEqual(abst_res.splits[3], [0])

    def test_split_by(self):

        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=3)
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=4)

        x = np.random.randn(10, 10, 10)
        abst_x1 = Abstraction().load(conf1, 'x1', x.shape, 'FLOAT', x)
        abst_x2 = Abstraction().load(conf2, 'x2', x.shape, 'FLOAT', x)

        abst_y1 = abst_x1.split_by([[0], list(range(10)), [0]], inplace=False)
        abst_y2 = abst_x1.split_by([list(range(10)), [0], [0]], inplace=False)

        self.assertTrue(correct_abstraction(abst_y1, x))
        self.assertTrue(correct_abstraction(abst_y2, x))

    def test_Conv(self):
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=[1, 3, 10, 10])
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=[4, 4, 5, 5])
        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3])

        X = np.random.randn(1, 10, 400, 400)
        W = np.random.randn(16, 5, 3, 3)
        B = np.random.randn(16) * 10.

        aX = Abstraction().load(conf1, 'X', X.shape, 'FLOAT', X)
        aW = Abstraction().load(conf2, 'W', W.shape, 'FLOAT', W)
        aB = Abstraction().load(conf3, 'B', B.shape, 'FLOAT', B)

        conv_node1 = helper.make_node(
            "Conv", ['X', 'W', 'b'], ['res'], "Conv",
            auto_pad='NOTSET', dilations=[1, 1], strides=[2, 2], group=2, pads=[100, 50, 100, 50]
        )

        res = torch.nn.functional.conv2d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(2, 2),
                                         padding=(100, 50), dilation=1, groups=2)
        aRes, _ = interp.interp_Conv([aX, aW, aB], conv_node1, 'Conv', 'res')
        self.assertTrue(correct_abstraction(aRes, res))

        # =======

        conv_node2 = helper.make_node(
            "Conv", ['X', 'W', 'b'], ['res'], "Conv",
            auto_pad='VALID', dilations=[3, 3], strides=[2, 2], group=2
        )
        aRes, _ = interp.interp_Conv([aX, aW, aB], conv_node2, 'Conv', 'res')
        res = torch.nn.functional.conv2d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(2, 2), padding=0,
                                         dilation=3, groups=2)
        self.assertTrue(correct_abstraction(aRes, res))

        # =======

        conv_node3 = helper.make_node(
            "Conv", ['X', 'W', 'b'], ['res'], "Conv",
            auto_pad='VALID', dilations=[3, 5], strides=[2, 5], group=2
        )
        aRes, _ = interp.interp_Conv([aX, aW, aB], conv_node3, 'Conv', 'res')
        res = torch.nn.functional.conv2d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(2, 5), padding=0,
                                         dilation=(3, 5), groups=2)
        self.assertTrue(correct_abstraction(aRes, res))

        # =======

        X = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(np.float32)
        W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        # Convolution with auto_pad='SAME_LOWER' and strides=2
        conv_node4 = helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            auto_pad='SAME_LOWER',
            kernel_shape=[3, 3],
            strides=[2, 2],
        )
        y = np.array([[[[12., 27., 24.],
                        [63., 108., 81.],
                        [72., 117., 84.]]]]).astype(np.float32)
        aX = Abstraction().load(conf1, 'X', X.shape, 'FLOAT', X)
        aW = Abstraction().load(conf2, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_Conv([aX, aW], conv_node4, 'Conv', 'res')
        self.assertTrue(correct_abstraction(aRes, y))

        # =======
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=[1, 3, 10,])
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=[4, 4, 3])
        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[5])

        X = np.random.randn(1, 5, 40)
        W = np.random.randn(16, 5, 4)
        B = np.random.randn(16) * 10.

        aX = Abstraction().load(conf1, 'X', X.shape, 'FLOAT', X)
        aW = Abstraction().load(conf2, 'W', W.shape, 'FLOAT', W)
        aB = Abstraction().load(conf3, 'B', B.shape, 'FLOAT', B)

        conv_node5 = helper.make_node(
            'Conv',
            inputs=['x', 'W', 'b'],
            outputs=['y'],
            pads=[4, 4],
            strides=[3],  # Default values for other attributes: dilations=[1, 1, 1], groups=1
        )
        res = torch.nn.functional.conv1d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(3),
                                         padding=[4])
        aRes, _ = interp.interp_Conv([aX, aW, aB], conv_node5, 'Conv', 'res')
        self.assertTrue(correct_abstraction(aRes, res))

        # =======
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=[1, 3, 10, 10, 10])
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=[4, 4, 5, 5, 3])
        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3])

        X = np.random.randn(1, 5, 40, 40, 101)
        W = np.random.randn(16, 5, 3, 3, 4)
        B = np.random.randn(16) * 10.

        aX = Abstraction().load(conf1, 'X', X.shape, 'FLOAT', X)
        aW = Abstraction().load(conf2, 'W', W.shape, 'FLOAT', W)
        aB = Abstraction().load(conf3, 'B', B.shape, 'FLOAT', B)

        conv_node6 = helper.make_node(
            'Conv',
            inputs=['x', 'W', 'b'],
            outputs=['y'],
            pads=[1, 0, 1, 1, 0, 1],
            strides=[2, 2, 3],  # Default values for other attributes: dilations=[1, 1, 1], groups=1
        )
        res = torch.nn.functional.conv3d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(2, 2, 3),
                                         padding=[1, 0, 1])
        aRes, _ = interp.interp_Conv([aX, aW, aB], conv_node6, 'Conv', 'res')
        self.assertTrue(correct_abstraction(aRes, res))

    def test_conv(self):
        pass


if __name__ == '__main__':
    unittest.main()
