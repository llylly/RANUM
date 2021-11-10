import unittest
import torch
import numpy as np
from onnx import helper, TensorProto
from onnx.backend.test.case.node.pool_op_common import get_output_shape, pool, get_pad_shape
from functools import reduce, partial

from interp.interp_utils import AbstractionInitConfig, EPS, PossibleNumericalError
from interp.interp_operator import Abstraction, Interpreter


def summary(obj: Abstraction):
    obj.print()


def tf_equal(a, b):
    # tensor float equal
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    sum_err = np.linalg.norm((a - np.array(b)).reshape(-1), ord=1)
    if sum_err > EPS:
        print('equal err:', sum_err)
    return sum_err <= EPS


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

        self.assertTrue(tf_equal(abst2.lb, [[-9.9, 5-9.9], [50-9.9, 55-9.9]]))
        self.assertTrue(tf_equal(abst2.ub, [[44+9.9, 49+9.9], [94+9.9, 99+9.9]]))

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

        self.assertTrue(correct_abstraction(abst_y, x.reshape((20, 10, 1, 1, 5))))

        node = helper.make_node(
            'Unsqueeze', ['v1'], ['s'], 'unsqueeze', axes=axes
        )

        abst_y_new, _ = interp.interp_Unsqueeze([abst_x, abst_axes], node, 'Unsqueeze', 'y')

        self.assertTrue(correct_abstraction(abst_y_new, x.reshape((20, 10, 1, 1, 5))))

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

    def test_SplitOp(self):
        interp = Interpreter()
        conf_empty = AbstractionInitConfig(diff=True, from_init=False)
        conf_s4 = AbstractionInitConfig(diff=True, from_init=True, stride=4)
        conf_s3 = AbstractionInitConfig(diff=True, from_init=True, stride=3)
        conf_s1 = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        conf_ind = AbstractionInitConfig(diff=False, from_init=True, stride=1)
        conf_ind2 = AbstractionInitConfig(diff=False, from_init=True, stride=2)

        input = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)

        node = helper.make_node(
            'Split',
            inputs=['input'],
            outputs=['output_1', 'output_2', 'output_3'],
            axis=0
        )

        expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4.]).astype(np.float32), np.array([5., 6.]).astype(np.float32)]

        a_input = Abstraction().load(conf_s3, 'input', input.shape, 'FLOAT', input)
        a_out, _ = interp.interp_Split([a_input], node, 'Split', 'output')
        self.assertTrue(correct_abstraction(a_out[0], expected_outputs[0]))
        self.assertTrue(correct_abstraction(a_out[1], expected_outputs[1]))
        self.assertTrue(correct_abstraction(a_out[2], expected_outputs[2]))

        split = np.array([2, 4]).astype(np.int64)
        a_split = Abstraction().load(conf_ind, 'split', split.shape, 'FLOAT', split)
        a_out, _ = interp.interp_Split([a_input, a_split], node, 'Split', 'output')

        expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4., 5., 6.]).astype(np.float32)]
        self.assertTrue(correct_abstraction(a_out[0], expected_outputs[0]))
        self.assertTrue(correct_abstraction(a_out[1], expected_outputs[1]))


        input = np.array([[1., 2., 3., 4., 5., 6.],
                          [7., 8., 9., 10., 11., 12.]]).astype(np.float32)

        node = helper.make_node(
            'Split',
            inputs=['input'],
            outputs=['output_1', 'output_2'],
            axis=1
        )

        expected_outputs = [np.array([[1., 2., 3.], [7., 8., 9.]]).astype(np.float32),
                            np.array([[4., 5., 6.], [10., 11., 12.]]).astype(np.float32)]

        a_input = Abstraction().load(conf_s3, 'input', input.shape, 'FLOAT', input)
        a_out, _ = interp.interp_Split([a_input], node, 'Split', 'output')
        self.assertTrue(correct_abstraction(a_out[0], expected_outputs[0]))
        self.assertTrue(correct_abstraction(a_out[1], expected_outputs[1]))

        a_input = Abstraction().load(conf_s1, 'input', input.shape, 'FLOAT', input)
        a_out, _ = interp.interp_Split([a_input], node, 'Split', 'output')
        self.assertTrue(correct_abstraction(a_out[0], expected_outputs[0], tight=True))
        self.assertTrue(correct_abstraction(a_out[1], expected_outputs[1], tight=True))

        split = np.array([2, 4]).astype(np.int64)
        node = helper.make_node(
            'Split',
            inputs=['input', 'split'],
            outputs=['output_1', 'output_2'],
            axis=1,
        )

        a_split = Abstraction().load(conf_ind, 'split', split.shape, 'FLOAT', split)
        a_input = Abstraction().load(conf_s3, 'input', input.shape, 'FLOAT', input)
        a_out, _ = interp.interp_Split([a_input], node, 'Split', 'output')
        self.assertTrue(correct_abstraction(a_out[0], expected_outputs[0]))
        self.assertTrue(correct_abstraction(a_out[1], expected_outputs[1]))

        node = helper.make_node(
            'Split',
            inputs=['input', 'split'],
            outputs=['output_1', 'output_2', 'output_3'],
            axis=0,
        )

        a_input = Abstraction().load(conf_empty, 'input', [0], 'FLOAT', None)
        a_out, _ = interp.interp_Split([a_input], node, 'Split', 'output')
        for i in range(3):
            self.assertListEqual(a_out[i].shape, [0])
            self.assertListEqual(a_out[i].splits, [[]])

        a_input = Abstraction().load(conf_s3, 'input', input.shape, 'FLOAT', input)
        a_out, _ = interp.interp_Split([a_input], node, 'Split', 'output')
        for i in range(3):
            if i < 2:
                self.assertListEqual(a_out[i].shape, [0, 6])
            else:
                self.assertListEqual(a_out[i].shape, [2, 6])
            if i < 2:
                self.assertListEqual(a_out[i].splits, [[], [0, 3]])
            else:
                self.assertListEqual(a_out[i].splits, [[0], [0, 3]])



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

    def test_RandomNormal(self):
        interp = Interpreter()
        node = helper.make_node(
            'RandomNormal', [], ['res'], 'RandomNormal',
            mean = 10.,
            scale = 3.,
            shape = [10, 4, 6]
        )
        abst_res, _ = interp.interp_RandomNormal([], node, 'RandomNormal', 'res')
        # abst_res.print()
        self.assertTrue(correct_abstraction(abst_res, np.full((10, 4, 6), 10.)))
        succeed = sum([int(correct_abstraction(abst_res, np.random.normal(10., 3., (10, 4, 6)))) for _ in range(1000)])
        self.assertTrue(succeed > 900)


        node = helper.make_node(
            'RandomNormal', [], ['res'], 'RandomNormal',
            mean = 10.,
            scale = 3.,
            shape = [10, 0, 6]
        )
        abst_res, _ = interp.interp_RandomNormal([], node, 'RandomNormal', 'res')
        # abst_res.print()
        self.assertTrue(abst_res.lb.numel() == 0)

    def test_BoolOps(self):
        for stride in [1, 2]:
            interp = Interpreter()
            x = np.random.randn(10, 20, 10, 30).astype(np.float64)
            conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=stride)
            abst_x = Abstraction().load(conf1, 'x', [10, 20, 10, 30], 'FLOAT', x)
            y = np.random.randn(1, 10, 1).astype(np.float64)
            conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=stride)
            abst_y = Abstraction().load(conf2, 'y', [1, 10, 1], 'FLOAT', y)
            ops = [lambda x, y: x < y, lambda x, y: x <= y, lambda x, y: x > y, lambda x, y: x >= y, lambda x, y: x == y]
            op_names = ["Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal"]
            op_interps = [interp.interp_Less, interp.interp_LessOrEqual, interp.interp_Greater,
                          interp.interp_GreaterOrEqual, interp.interp_Equal]

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

    def test_Clip(self):
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=1)

        node = helper.make_node(
            'Clip',
            inputs=['x', 'min', 'max'],
            outputs=['y'],
        )

        x = np.array([-2, 0, 2]).astype(np.float32)
        min_val = np.float32(-1)
        max_val = np.float32(1)
        y = np.clip(x, min_val, max_val)  # expected output [-1., 0., 1.]

        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_min_val = Abstraction().load(conf1, 'min', min_val.shape, 'FLOAT', min_val)
        a_max_val = Abstraction().load(conf1, 'max', max_val.shape, 'FLOAT', max_val)
        # expect(node, inputs=[x, min_val, max_val], outputs=[y],
        #        name='test_clip_example')
        a_y, _ = interp.interp_Clip([a_x, a_min_val, a_max_val], node, 'Clip', 'y')
        self.assertTrue(correct_abstraction(a_y, y))

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, min_val, max_val)
        # expect(node, inputs=[x, min_val, max_val], outputs=[y],
        #        name='test_clip')
        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_Clip([a_x, a_min_val, a_max_val], node, 'Clip', 'y')
        self.assertTrue(correct_abstraction(a_y, y))

        node = helper.make_node(
            'Clip',
            inputs=['x', 'min', 'max'],
            outputs=['y'],
        )

        min_val = np.float32(-5)
        max_val = np.float32(5)

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        # expect(node, inputs=[x, min_val, max_val], outputs=[y],
        #        name='test_clip_inbounds')
        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_min_val = Abstraction().load(conf1, 'min', min_val.shape, 'FLOAT', min_val)
        a_max_val = Abstraction().load(conf1, 'max', max_val.shape, 'FLOAT', max_val)
        a_y, _ = interp.interp_Clip([a_x, a_min_val, a_max_val], node, 'Clip', 'y')
        self.assertTrue(correct_abstraction(a_y, y))

        x = np.array([-6, 0, 6]).astype(np.float32)
        y = np.array([-5, 0, 5]).astype(np.float32)
        # expect(node, inputs=[x, min_val, max_val], outputs=[y],
        #        name='test_clip_outbounds')
        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_min_val = Abstraction().load(conf1, 'min', min_val.shape, 'FLOAT', min_val)
        a_max_val = Abstraction().load(conf1, 'max', max_val.shape, 'FLOAT', max_val)
        a_y, _ = interp.interp_Clip([a_x, a_min_val, a_max_val], node, 'Clip', 'y')
        self.assertTrue(correct_abstraction(a_y, y))

        x = np.array([-1, 0, 6]).astype(np.float32)
        y = np.array([-1, 0, 5]).astype(np.float32)
        # expect(node, inputs=[x, min_val, max_val], outputs=[y],
        #        name='test_clip_splitbounds')
        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_min_val = Abstraction().load(conf1, 'min', min_val.shape, 'FLOAT', min_val)
        a_max_val = Abstraction().load(conf1, 'max', max_val.shape, 'FLOAT', max_val)
        a_y, _ = interp.interp_Clip([a_x, a_min_val, a_max_val], node, 'Clip', 'y')
        self.assertTrue(correct_abstraction(a_y, y))

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

    def test_LogSoftmax(self):
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=2)

        node = helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output
        # [[-2.4076061 -1.407606  -0.407606 ]]
        y = logsoftmax(x)

        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=True))

        a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=False))

        # =======

        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]
                     ).astype(np.float32)
        # expected output
        # [[-3.4401896  -2.4401896  -1.4401896  -0.44018966]
        # [-3.4401896  -2.4401896  -1.4401896  -0.44018966]]
        y = logsoftmax(x)

        node = helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
        )
        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=True))

        a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        a_y, exceps = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        self.assertTrue(len(exceps) > 0)
        # self.assertTrue(correct_abstraction(a_y, y, tight=False))

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        node = helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
            axis=0,
        )
        y = logsoftmax(x, axis=0)

        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=True))

        a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=False))

        node = helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
            axis=1,
        )
        y = logsoftmax(x, axis=1)

        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=True))

        a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=False))

        node = helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
            axis=2,
        )
        y = logsoftmax(x, axis=2)

        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=True))

        a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=False))

        node = helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
            axis=-1,
        )
        y = logsoftmax(x, axis=-1)

        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=True))

        a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=False))

        # default axis is -1
        node = helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
        )

        a_x = Abstraction().load(conf1, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=True))

        a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        a_y, _ = interp.interp_LogSoftmax([a_x], node, 'LogSoftmax', 'y')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y, tight=False))

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
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=[1, 3, 10, ])
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


        # =======
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=[1, 3, 10, 10, 10])
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=[4, 4, 5, 5, 3])
        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3])

        X = np.random.randn(1, 5, 40, 40, 101)
        W = np.random.randn(5, 1, 3, 3, 4)
        B = np.random.randn(5) * 10.

        aX = Abstraction().load(conf1, 'X', X.shape, 'FLOAT', X)
        aW = Abstraction().load(conf2, 'W', W.shape, 'FLOAT', W)
        aB = Abstraction().load(conf3, 'B', B.shape, 'FLOAT', B)

        conv_node6 = helper.make_node(
            'Conv',
            inputs=['x', 'W', 'b'],
            outputs=['y'],
            pads=[1, 0, 1, 1, 0, 1],
            strides=[2, 2, 3],  # Default values for other attributes: dilations=[1, 1, 1], groups=1
            group=5
        )
        res = torch.nn.functional.conv3d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(2, 2, 3),
                                         padding=[1, 0, 1], groups=5)
        aRes, _ = interp.interp_Conv([aX, aW, aB], conv_node6, 'Conv', 'res')
        self.assertTrue(correct_abstraction(aRes, res))

    def test_ConvTranspose(self):

        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=[1, 3, 10, 10])
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=[4, 4, 5, 5])
        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3])
        conf_s3 = AbstractionInitConfig(diff=True, from_init=True, stride=3)
        conf_s2 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        conf_s1 = AbstractionInitConfig(diff=True, from_init=True, stride=1)

        X = np.random.randn(1, 10, 400, 400)
        W = np.random.randn(10, 8, 3, 3)
        B = np.random.randn(16) * 10.

        aX = Abstraction().load(conf1, 'X', X.shape, 'FLOAT', X)
        aW = Abstraction().load(conf2, 'W', W.shape, 'FLOAT', W)
        aB = Abstraction().load(conf3, 'B', B.shape, 'FLOAT', B)

        conv_node1 = helper.make_node(
            "ConvTranspose", ['X', 'W', 'b'], ['res'], "ConvTranspose",
            auto_pad='NOTSET', dilations=[1, 1], strides=[2, 2], group=2, pads=[2, 10, 2, 10]
        )

        res = torch.nn.functional.conv_transpose2d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(2, 2),
                                         padding=(2, 10), dilation=1, groups=2)
        aRes, _ = interp.interp_ConvTranspose([aX, aW, aB], conv_node1, 'Conv', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, res))
        # print('chkp1')

        # =======

        conv_node2 = helper.make_node(
            "ConvTranspose", ['X', 'W', 'b'], ['res'], "ConvTranspose",
            auto_pad='VALID', dilations=[3, 3], strides=[2, 2], group=2
        )
        aRes, _ = interp.interp_ConvTranspose([aX, aW, aB], conv_node2, 'ConvTranspose', 'res')
        res = torch.nn.functional.conv_transpose2d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(2, 2), padding=0,
                                         dilation=3, groups=2)
        self.assertTrue(correct_abstraction(aRes, res))
        # print('chkp2')

        # =======

        conv_node3 = helper.make_node(
            "ConvTranspose", ['X', 'W', 'b'], ['res'], "ConvTranspose",
            auto_pad='VALID', dilations=[3, 5], strides=[2, 5], group=2
        )
        aRes, _ = interp.interp_ConvTranspose([aX, aW, aB], conv_node3, 'ConvTranspose', 'res')
        res = torch.nn.functional.conv_transpose2d(torch.tensor(X), torch.tensor(W), torch.tensor(B), stride=(2, 5), padding=0,
                                         dilation=(3, 5), groups=2)
        self.assertTrue(correct_abstraction(aRes, res))
        # print('chkp3')

        # =======

        x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                        [3., 4., 5.],
                        [6., 7., 8.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                        [1., 1., 1.],
                        [1., 1., 1.]],
                       [[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s2, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()

        y = np.array([[[[0., 1., 3., 3., 2.],  # (1, 2, 5, 5)
                        [3., 8., 15., 12., 7.],
                        [9., 21., 36., 27., 15.],
                        [9., 20., 33., 24., 13.],
                        [6., 13., 21., 15., 8.]],

                       [[0., 1., 3., 3., 2.],
                        [3., 8., 15., 12., 7.],
                        [9., 21., 36., 27., 15.],
                        [9., 20., 33., 24., 13.],
                        [6., 13., 21., 15., 8.]]]]).astype(np.float32)
        self.assertTrue(correct_abstraction(aRes, y, tight=True))


        aX = Abstraction().load(conf_s2, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s2, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=False))
        # print('chkp4')

        # =======

        x = np.array([[[0., 1., 2.]]]).astype(np.float32)  # (1, 1, 3)

        W = np.array([[[1., 1., 1.],  # (1, 2, 3)
                       [1., 1., 1.]]]).astype(np.float32)

        node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s2, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()

        y = np.array([[[0., 1., 3., 3., 2.],  # (1, 2, 5)
                       [0., 1., 3., 3., 2.]]]).astype(np.float32)
        self.assertTrue(correct_abstraction(aRes, y, tight=True))

        aX = Abstraction().load(conf_s2, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s2, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=False))
        # print('chkp5')

        # =======

        x = np.array([[[[[0., 1., 2., 3., 4.],  # (1, 1, 3, 4, 5)
                         [5., 6., 7., 8., 9.],
                         [10., 11., 12., 13., 14.],
                         [15., 16., 17., 18., 19.]],
                        [[20., 21., 22., 23., 24.],
                         [25., 26., 27., 28., 29.],
                         [30., 31., 32., 33., 34.],
                         [35., 36., 37., 38., 39.]],
                        [[40., 41., 42., 43., 44.],
                         [45., 46., 47., 48., 49.],
                         [50., 51., 52., 53., 54.],
                         [55., 56., 57., 58., 59.]]]]]).astype(np.float32)

        W = np.array([[[[[1., 1., 1.],  # (1, 2, 3, 3, 3)
                         [1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]]],
                       [[[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]]]]]).astype(np.float32)

        y = np.array([[[[[0., 1., 3., 6., 9., 7., 4.],  # (1, 2, 5, 6, 7)
                 [5., 12., 21., 27., 33., 24., 13.],
                 [15., 33., 54., 63., 72., 51., 27.],
                 [30., 63., 99., 108., 117., 81., 42.],
                 [25., 52., 81., 87., 93., 64., 33.],
                 [15., 31., 48., 51., 54., 37., 19.]],

                [[20., 42., 66., 72., 78., 54., 28.],
                 [50., 104., 162., 174., 186., 128., 66.],
                 [90., 186., 288., 306., 324., 222., 114.],
                 [120., 246., 378., 396., 414., 282., 144.],
                 [90., 184., 282., 294., 306., 208., 106.],
                 [50., 102., 156., 162., 168., 114., 58.]],

                [[60., 123., 189., 198., 207., 141., 72.],
                 [135., 276., 423., 441., 459., 312., 159.],
                 [225., 459., 702., 729., 756., 513., 261.],
                 [270., 549., 837., 864., 891., 603., 306.],
                 [195., 396., 603., 621., 639., 432., 219.],
                 [105., 213., 324., 333., 342., 231., 117.]],

                [[60., 122., 186., 192., 198., 134., 68.],
                 [130., 264., 402., 414., 426., 288., 146.],
                 [210., 426., 648., 666., 684., 462., 234.],
                 [240., 486., 738., 756., 774., 522., 264.],
                 [170., 344., 522., 534., 546., 368., 186.],
                 [90., 182., 276., 282., 288., 194., 98.]],

                [[40., 81., 123., 126., 129., 87., 44.],
                 [85., 172., 261., 267., 273., 184., 93.],
                 [135., 273., 414., 423., 432., 291., 147.],
                 [150., 303., 459., 468., 477., 321., 162.],
                 [105., 212., 321., 327., 333., 224., 113.],
                 [55., 111., 168., 171., 174., 117., 59.]]],

               [[[0., 1., 3., 6., 9., 7., 4.],
                 [5., 12., 21., 27., 33., 24., 13.],
                 [15., 33., 54., 63., 72., 51., 27.],
                 [30., 63., 99., 108., 117., 81., 42.],
                 [25., 52., 81., 87., 93., 64., 33.],
                 [15., 31., 48., 51., 54., 37., 19.]],

                [[20., 42., 66., 72., 78., 54., 28.],
                 [50., 104., 162., 174., 186., 128., 66.],
                 [90., 186., 288., 306., 324., 222., 114.],
                 [120., 246., 378., 396., 414., 282., 144.],
                 [90., 184., 282., 294., 306., 208., 106.],
                 [50., 102., 156., 162., 168., 114., 58.]],

                [[60., 123., 189., 198., 207., 141., 72.],
                 [135., 276., 423., 441., 459., 312., 159.],
                 [225., 459., 702., 729., 756., 513., 261.],
                 [270., 549., 837., 864., 891., 603., 306.],
                 [195., 396., 603., 621., 639., 432., 219.],
                 [105., 213., 324., 333., 342., 231., 117.]],

                [[60., 122., 186., 192., 198., 134., 68.],
                 [130., 264., 402., 414., 426., 288., 146.],
                 [210., 426., 648., 666., 684., 462., 234.],
                 [240., 486., 738., 756., 774., 522., 264.],
                 [170., 344., 522., 534., 546., 368., 186.],
                 [90., 182., 276., 282., 288., 194., 98.]],

                [[40., 81., 123., 126., 129., 87., 44.],
                 [85., 172., 261., 267., 273., 184., 93.],
                 [135., 273., 414., 423., 432., 291., 147.],
                 [150., 303., 459., 468., 477., 321., 162.],
                 [105., 212., 321., 327., 333., 224., 113.],
                 [55., 111., 168., 171., 174., 117., 59.]]]]]).astype(np.float32)

        node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s2, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=True))

        aX = Abstraction().load(conf_s2, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s3, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y))

        aX = Abstraction().load(conf_s3, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s3, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y))

        # print('chkp6')

        # =======

        x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                        [3., 4., 5.],
                        [6., 7., 8.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                        [1., 1., 1.],
                        [1., 1., 1.]],
                       [[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        y = np.array([[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0.]],

                       [[0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(np.float32)


        node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                             strides=[3, 2],
                             output_shape=[10, 8])

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s1, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=True))


        aX = Abstraction().load(conf_s3, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s3, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=False))


        node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                     strides=[3, 2],
                                     output_padding=[1, 1])

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s1, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=True))


        aX = Abstraction().load(conf_s3, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s3, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=False))

        node = helper.make_node(
            'ConvTranspose', ['X', 'W'], ['Y'],
            name='test',
            strides=[3, 2],
            output_shape=[10, 8],
            kernel_shape=[3, 3],
            output_padding=[1, 1]
        )

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s1, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=True))

        aX = Abstraction().load(conf_s3, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s3, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=False))

        # print('chkp7')

        # =======

        x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                        [3., 4., 5.],
                        [6., 7., 8.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                        [1., 1., 1.],
                        [1., 1., 1.]],
                       [[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"], auto_pad="SAME_UPPER", strides=[2, 2])

        y = np.array([[[[0., 0., 1., 1., 3., 2.],
                        [0., 0., 1., 1., 3., 2.],
                        [3., 3., 8., 5., 12., 7.],
                        [3., 3., 7., 4., 9., 5.],
                        [9., 9., 20., 11., 24., 13.],
                        [6., 6., 13., 7., 15., 8.]],

                       [[0., 0., 1., 1., 3., 2.],
                        [0., 0., 1., 1., 3., 2.],
                        [3., 3., 8., 5., 12., 7.],
                        [3., 3., 7., 4., 9., 5.],
                        [9., 9., 20., 11., 24., 13.],
                        [6., 6., 13., 7., 15., 8.]]]]).astype(np.float32)

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s1, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=True))


        aX = Abstraction().load(conf_s3, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s3, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=False))

        # print('chkp8')

        # =======

        x = np.array([[[[3., 8., 1.],  # (1, 1, 3, 3)
                        [9., 5., 7.],
                        [3., 2., 6.]]]]).astype(np.float32)
        W = np.array([[[[7., 2.],  # (1, 1, 2, 2)
                        [1., 9.]]]]).astype(np.float32)

        node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"], dilations=[2, 2])

        y = np.array([[[[21., 56., 13., 16., 2.],  # [1, 1, 5, 5]
                        [63., 35., 67., 10., 14.],
                        [24., 22., 76., 76., 21.],
                        [9., 5., 88., 45., 63.],
                        [3., 2., 33., 18., 54.]]]]).astype(np.float32)

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s1, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=True))


        aX = Abstraction().load(conf_s3, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s3, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=False))

        # print('chkp9')

        # ========

        x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                        [3., 4., 5.],
                        [6., 7., 8.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                        [1., 1., 1.],
                        [1., 1., 1.]],
                       [[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                     strides=[3, 2],
                                     pads=[1, 2, 1, 2])

        y = np.array([[[[1., 1., 3.],  # (1, 2, 7, 3)
                        [1., 1., 3.],
                        [7., 4., 9.],
                        [7., 4., 9.],
                        [7., 4., 9.],
                        [13., 7., 15.],
                        [13., 7., 15.]],

                       [[1., 1., 3.],
                        [1., 1., 3.],
                        [7., 4., 9.],
                        [7., 4., 9.],
                        [7., 4., 9.],
                        [13., 7., 15.],
                        [13., 7., 15.]]]]).astype(np.float32)

        aX = Abstraction().load(conf_s1, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s1, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=True))


        aX = Abstraction().load(conf_s3, 'x', x.shape, 'FLOAT', x)
        aW = Abstraction().load(conf_s3, 'W', W.shape, 'FLOAT', W)
        aRes, _ = interp.interp_ConvTranspose([aX, aW], node, 'ConvTranspose', 'res')
        # aRes.print()
        self.assertTrue(correct_abstraction(aRes, y, tight=False))

        # print('chkp10')

    def test_Pad(self):
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=3)

        data = np.array(
            [
                [1.0, 1.2],
                [2.3, 3.4],
                [4.5, 5.7],
            ])
        pads = np.array([0, 2, 0, 0])
        constant_value = np.array(0.0)

        y = np.array(
            [
                [0.0, 0.0, 1.0, 1.2],
                [0.0, 0.0, 2.3, 3.4],
                [0.0, 0.0, 4.5, 5.7],
            ]
        )

        node = helper.make_node(
            "Pad", ['data', 'pads', 'constant_value'], ['res'],
            mode='constant'
        )

        a_data = Abstraction().load(conf2, 'data', data.shape, 'FLOAT', data)
        a_pads = Abstraction().load(conf1, 'pads', pads.shape, 'FLOAT', pads)
        a_constant_value = Abstraction().load(conf1, 'constant_value', constant_value.shape, 'FLOAT', constant_value)
        a_res, _ = interp.interp_Pad([a_data, a_pads, a_constant_value], node, 'Pad', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, y))

        a_data = Abstraction().load(conf1, 'data', data.shape, 'FLOAT', data)
        a_pads = Abstraction().load(conf1, 'pads', pads.shape, 'FLOAT', pads)
        a_res, _ = interp.interp_Pad([a_data, a_pads, a_constant_value], node, 'Pad', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, y, tight=True))

        # =======

        y = np.array(
            [
                [1.0, 1.2, 1.0, 1.2],
                [2.3, 3.4, 2.3, 3.4],
                [4.5, 5.7, 4.5, 5.7],
            ]
        )

        node = helper.make_node(
            "Pad", ['data', 'pads', 'constant_value'], ['res'],
            mode='reflect'
        )

        a_data = Abstraction().load(conf2, 'data', data.shape, 'FLOAT', data)
        a_pads = Abstraction().load(conf1, 'pads', pads.shape, 'FLOAT', pads)
        a_constant_value = Abstraction().load(conf1, 'constant_value', constant_value.shape, 'FLOAT', constant_value)
        a_res, _ = interp.interp_Pad([a_data, a_pads, a_constant_value], node, 'Pad', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, y))

        a_data = Abstraction().load(conf1, 'data', data.shape, 'FLOAT', data)
        a_res, _ = interp.interp_Pad([a_data, a_pads, a_constant_value], node, 'Pad', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, y, tight=True))

        # =======

        y = np.array(
            [
                [1.0, 1.0, 1.0, 1.2],
                [2.3, 2.3, 2.3, 3.4],
                [4.5, 4.5, 4.5, 5.7],
            ]
        )

        node = helper.make_node(
            "Pad", ['data', 'pads', 'constant_value'], ['res'],
            mode='edge'
        )

        a_data = Abstraction().load(conf2, 'data', data.shape, 'FLOAT', data)
        a_pads = Abstraction().load(conf1, 'pads', pads.shape, 'FLOAT', pads)
        a_constant_value = Abstraction().load(conf1, 'constant_value', constant_value.shape, 'FLOAT', constant_value)
        a_res, _ = interp.interp_Pad([a_data, a_pads, a_constant_value], node, 'Pad', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, y))

        a_data = Abstraction().load(conf1, 'data', data.shape, 'FLOAT', data)
        a_res, _ = interp.interp_Pad([a_data, a_pads, a_constant_value], node, 'Pad', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, y, tight=True))

        # =======

        node = helper.make_node(
            'Pad',
            inputs=['x', 'pads', 'value'],
            outputs=['y'],
            mode='constant'
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.float32(1.2)

        y = pad_impl(
            x,
            pads,
            'constant',
            1.2
        )
        a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
        a_pads = Abstraction().load(conf1, 'pads', pads.shape, 'FLOAT', pads)
        a_value = Abstraction().load(conf1, 'value', value.shape, 'FLOAT', value)
        a_y, _ = interp.interp_Pad([a_x, a_pads, a_value], node, 'Pad', 'res')
        # a_y.print()
        self.assertTrue(correct_abstraction(a_y, y))

        # =======

        for mode in ['edge', 'reflect']:
            # print(mode)
            node = helper.make_node(
                'Pad',
                inputs=['x', 'pads'],
                outputs=['y'],
                mode=mode
            )
            x = np.random.randn(1, 3, 4, 5).astype(np.int32)
            pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            y = pad_impl(
                x,
                pads,
                mode
            )
            # print(y)

            a_x = Abstraction().load(conf2, 'x', x.shape, 'FLOAT', x)
            a_pads = Abstraction().load(conf1, 'pads', pads.shape, 'FLOAT', pads)
            a_y, _ = interp.interp_Pad([a_x, a_pads], node, 'Pad', 'res')
            # a_y.print()
            self.assertTrue(correct_abstraction(a_y, y))



    def test_MaxPool(self):
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=[1, 3, 2])
        conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=[2, 4, 5, 5])
        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3, 5, 10, 10, 10])

        def check(conf, node, x, y):
            ax = Abstraction().load(conf, 'X', x.shape, 'FLOAT', x)
            aRes, _ = interp.interp_MaxPool([ax], node, 'MaxPool', 'res')
            self.assertTrue(correct_abstraction(aRes, y))

        # maxpool_1d_default
        pool_node1 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2],
        )
        x = np.random.randn(1, 3, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = [2]
        strides = [1]
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'MAX')
        check(conf1, pool_node1, x, y)

        # maxpool_2d_ceil
        pool_node2 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            strides=[2, 2],
            ceil_mode=True
        )
        x = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]]).astype(np.float32)
        y = np.array([[[
            [11, 12],
            [15, 16]]]]).astype(np.float32)

        check(conf2, pool_node2, x, y)

        # maxpool_2d_default
        pool_node3 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')

        check(conf2, pool_node3, x, y)

        # maxpool_2d_dilations
        pool_node4 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            strides=[1, 1],
            dilations=[2, 2]
        )
        x = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]]).astype(np.float32)
        y = np.array([[[
            [11, 12],
            [15, 16]]]]).astype(np.float32)

        check(conf2, pool_node4, x, y)

        # maxpool_2d_pads
        pool_node5 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[2, 2, 2, 2]
        )
        x = np.random.randn(1, 3, 28, 28).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = pad_top = pad_right = pad_left = 2
        pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
        out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
        padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                        constant_values=np.nan)
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

        check(conf2, pool_node5, x, y)

        # maxpool_2d_precomputed_pads
        pool_node6 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2]

        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[
            [13, 14, 15, 15, 15],
            [18, 19, 20, 20, 20],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25]]]]).astype(np.float32)

        check(conf2, pool_node6, x, y)

        # maxpool_2d_precomputed_same_upper
        pool_node7 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            strides=[2, 2],
            auto_pad='SAME_UPPER'
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9, 10],
                        [17, 19, 20],
                        [22, 24, 25]]]]).astype(np.float32)

        check(conf2, pool_node7, x, y)

        # maxpool_2d_precomputed_strides
        pool_node8 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            strides=[2, 2]
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9],
                        [17, 19]]]]).astype(np.float32)

        check(conf2, pool_node8, x, y)

        # maxpool_2d_same_lower
        pool_node9 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            auto_pad='SAME_LOWER'
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
        pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
        pad_bottom = pad_shape[0] // 2
        pad_top = pad_shape[0] - pad_bottom
        pad_right = pad_shape[1] // 2
        pad_left = pad_shape[1] - pad_right
        padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                        constant_values=np.nan)
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

        check(conf2, pool_node9, x, y)

        # maxpool_2d_same_upper
        pool_node10 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            auto_pad='SAME_UPPER'
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
        pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
        pad_top = pad_shape[0] // 2
        pad_bottom = pad_shape[0] - pad_top
        pad_left = pad_shape[1] // 2
        pad_right = pad_shape[1] - pad_left
        padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                        constant_values=np.nan)
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

        check(conf2, pool_node10, x, y)

        # maxpool_2d_strides
        pool_node11 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            strides=[3, 3]
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (5, 5)
        strides = (3, 3)
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')

        check(conf2, pool_node11, x, y)

        # maxpool_3d_default
        pool_node12 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2, 2],
        )
        x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = [2, 2, 2]
        strides = [1, 1, 1]
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'MAX')

        check(conf3, pool_node12, x, y)


        # maxpool_3d_check_indices
        pool_node12 = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y', 'ind'],
            kernel_shape=[2, 2, 2],
        )
        x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = [2, 2, 2]
        strides = [1, 1, 1]
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'MAX')

        # check(conf3, pool_node12, x, y)
        ax = Abstraction().load(conf3, 'X', x.shape, 'FLOAT', x)
        aResList, _ = interp.interp_MaxPool([ax], pool_node12, 'MaxPool', 'res')
        self.assertTrue(correct_abstraction(aResList[0], y))

        self.assertListEqual(aResList[1].shape, aResList[0].shape)

    def test_AveragePool(self):
        for mode in ['precise', 'coarse']:
            interp = Interpreter(average_pool_mode=mode)
            conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=[1, 2, 3])
            conf2 = AbstractionInitConfig(diff=True, from_init=True, stride=[2, 4, 4, 5])
            conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3, 2, 8, 9, 2])

            def check(conf, node, x, y):
                ax = Abstraction().load(conf, 'X', x.shape, 'FLOAT', x)
                aRes, _ = interp.interp_AveragePool([ax], node, 'AveragePool', 'res')
                self.assertTrue(correct_abstraction(aRes, y))

            # averagepool_1d_default
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2],
            )
            x = np.random.randn(1, 3, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = [2]
            strides = [1]
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'AVG')

            check(conf1, node, x, y)

            # averagepool_2d_ceil
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                strides=[2, 2],
                ceil_mode=True
            )
            x = np.array([[[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]]]).astype(np.float32)
            y = np.array([[[
                [6, 7.5],
                [12, 13.5]]]]).astype(np.float32)

            check(conf2, node, x, y)

            # averagepool_2d_default
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')

            check(conf2, node, x, y)

            # averagepool_2d_pads
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1]
            )
            x = np.random.randn(1, 3, 28, 28).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (3, 3)
            strides = (1, 1)
            pad_bottom = 1
            pad_top = 1
            pad_right = 1
            pad_left = 1
            pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
            out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

            check(conf2, node, x, y)

            # averagepool_2d_pads_count_include_pad
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[4, 4],
                pads=[2, 2, 2, 2],
                count_include_pad=1,
            )
            x = np.random.randn(1, 3, 28, 28).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (4, 4)
            strides = (1, 1)
            pad_bottom = 2
            pad_top = 2
            pad_right = 2
            pad_left = 2
            pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
            out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=0)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG', count_include_pad=1)

            check(conf2, node, x, y)

            # averagepool_2d_precomputed_pads
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2]

            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[7, 7.5, 8, 8.5, 9],
                            [9.5, 10, 10.5, 11, 11.5],
                            [12, 12.5, 13, 13.5, 14],
                            [14.5, 15, 15.5, 16, 16.5],
                            [17, 17.5, 18, 18.5, 19]]]]).astype(np.float32)

            check(conf2, node, x, y)

            # averagepool_2d_precomputed_pads_count_include_pad
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2],
                count_include_pad=1
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[2.5200, 3.6000, 4.8000, 4.0800, 3.2400],
                            [4.5600, 6.4000, 8.4000, 7.0400, 5.5200],
                            [7.2000, 10.0000, 13.0000, 10.8000, 8.4000],
                            [6.9600, 9.6000, 12.4000, 10.2400, 7.9200],
                            [6.1200, 8.4000, 10.8000, 8.8800, 6.8400]]]]).astype(np.float32)

            check(conf2, node, x, y)

            # averagepool_2d_precomputed_same_upper
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad='SAME_UPPER'
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[4, 5.5, 7],
                            [11.5, 13, 14.5],
                            [19, 20.5, 22]]]]).astype(np.float32)

            check(conf2, node, x, y)

            # averagepool_2d_precomputed_strides
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                strides=[2, 2]
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[4, 6],
                            [14, 16]]]]).astype(np.float32)

            check(conf2, node, x, y)

            # averagepool_2d_same_lower strides != 1
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad='SAME_LOWER'
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (3, 3)
            strides = (2, 2)
            out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_bottom = pad_shape[0] // 2
            pad_top = pad_shape[0] - pad_bottom
            pad_right = pad_shape[1] // 2
            pad_left = pad_shape[1] - pad_right
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

            check(conf2, node, x, y)

            # averagepool_2d_same_lower strides == 1
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                auto_pad='SAME_LOWER'
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_bottom = pad_shape[0] // 2
            pad_top = pad_shape[0] - pad_bottom
            pad_right = pad_shape[1] // 2
            pad_left = pad_shape[1] - pad_right
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

            check(conf2, node, x, y)

            # averagepool_2d_same_upper strides == 1
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                auto_pad='SAME_UPPER'
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_top = pad_shape[0] // 2
            pad_bottom = pad_shape[0] - pad_top
            pad_left = pad_shape[1] // 2
            pad_right = pad_shape[1] - pad_left
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

            if mode == 'precise':
                with self.assertRaises(NotImplementedError):
                    check(conf2, node, x, y)

                    # averagepool_2d_same_upper strides != 1
                    node = helper.make_node(
                        'AveragePool',
                        inputs=['x'],
                        outputs=['y'],
                        kernel_shape=[3, 3],
                        strides=[2, 2],
                        auto_pad='SAME_UPPER'
                    )
                    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
                    x_shape = np.shape(x)
                    kernel_shape = (3, 3)
                    strides = (2, 2)
                    out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
                    pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
                    pad_top = pad_shape[0] // 2
                    pad_bottom = pad_shape[0] - pad_top
                    pad_left = pad_shape[1] // 2
                    pad_right = pad_shape[1] - pad_left
                    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                                    constant_values=np.nan)
                    y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

                    with self.assertRaises(NotImplementedError):
                        check(conf2, node, x, y)
            else:
                check(conf2, node, x, y)

                # averagepool_2d_same_upper strides != 1
                node = helper.make_node(
                    'AveragePool',
                    inputs=['x'],
                    outputs=['y'],
                    kernel_shape=[3, 3],
                    strides=[2, 2],
                    auto_pad='SAME_UPPER'
                )
                x = np.random.randn(1, 3, 32, 32).astype(np.float32)
                x_shape = np.shape(x)
                kernel_shape = (3, 3)
                strides = (2, 2)
                out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
                pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
                pad_top = pad_shape[0] // 2
                pad_bottom = pad_shape[0] - pad_top
                pad_left = pad_shape[1] // 2
                pad_right = pad_shape[1] - pad_left
                padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                                constant_values=np.nan)
                y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

                check(conf2, node, x, y)

            # averagepool_2d_strides
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[5, 5],
                strides=[3, 3]
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (5, 5)
            strides = (3, 3)
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')

            check(conf2, node, x, y)

            # averagepool_3d_default
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2, 2],
            )
            x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = [2, 2, 2]
            strides = [1, 1, 1]
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'AVG')

            check(conf3, node, x, y)

            # averagepool_2d_same_lower_count_include_pad
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                auto_pad='SAME_LOWER',
                count_include_pad=1
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_bottom = pad_shape[0] // 2
            pad_top = pad_shape[0] - pad_bottom
            pad_right = pad_shape[1] // 2
            pad_left = pad_shape[1] - pad_right
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=0)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG', count_include_pad=1)

            check(conf2, node, x, y)

            # averagepool_2d_same_upper_count_include_pad strides != 1
            node = helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                auto_pad='SAME_UPPER',
                strides=[2, 2],
                count_include_pad=1
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (3, 3)
            strides = (2, 2)
            out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_top = pad_shape[0] // 2
            pad_bottom = pad_shape[0] - pad_top
            pad_left = pad_shape[1] // 2
            pad_right = pad_shape[1] - pad_left
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=0)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG', count_include_pad=1)

            check(conf2, node, x, y)

    def test_Flatten(self):
        """
            Test flatten
        :return:
        """
        interp = Interpreter()
        conf1 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        conf2 = AbstractionInitConfig(diff=False, from_init=True, stride=1)

        targ_shape = [5, 5, -1]

        a = np.array(list(range(5 * 5 * 5 * 5 * 2))).reshape((5, 5, 5, 5, 2))
        abst1 = Abstraction()
        abst1.load(conf1, 'v1', [5, 5, 5, 5, 2], 'FLOAT', a)


        node = helper.make_node(
            'Flatten',
            inputs=['x'],
            outputs=['y'],
            axis=2
        )

        b = a.reshape(tuple(targ_shape))
        abst2, _ = interp.interp_Flatten([abst1], node, 'Flatten', 'y')
        self.assertTrue(correct_abstraction(abst2, b))

        # =========

        targ_shape = [-1]
        c = a.reshape(tuple(targ_shape))

        node = helper.make_node(
            'Flatten',
            inputs=['x'],
            outputs=['y'],
            axis=0
        )
        abst3, _ = interp.interp_Flatten([abst1], node, 'Flatten', 'y')
        self.assertTrue(correct_abstraction(abst3, c))

        # summary(abst3)

        # ============

        conf3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3, 3, 3])

        targ_shape = [6, -1]

        a = np.array(list(range(6 * 6 * 6))).reshape((6, 6, 6))
        abst1 = Abstraction()
        abst1.load(conf3, 'v2', [6, 6, 6], 'FLOAT', a)

        d = a.reshape(tuple(targ_shape))
        node = helper.make_node(
            'Flatten',
            inputs=['x'],
            outputs=['y'],
        )
        abst4, _ = interp.interp_Flatten([abst1], node, 'Flatten', 'y')
        self.assertTrue(correct_abstraction(abst4, d))

        # summary(abst4)

    def test_Gemm(self):

        # all_attributes

        node = helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            alpha=0.25,
            beta=0.35,
            transA=1,
            transB=1
        )
        a = np.random.ranf([4, 3]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.random.ranf([1, 5]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)

        interp = Interpreter()
        conf12d = AbstractionInitConfig(diff=True, from_init=True, stride=[2, 2])
        conf11d = AbstractionInitConfig(diff=True, from_init=True, stride=[3])

        _a = Abstraction().load(conf12d, 'a', a.shape, 'FLOAT', a)
        _b = Abstraction().load(conf12d, 'b', b.shape, 'FLOAT', b)
        _c = Abstraction().load(conf12d, 'c', c.shape, 'FLOAT', c)
        abst_y, _ = interp.interp_Gemm([_a, _b, _c], node, 'Gemm', 'y')

        self.assertTrue(correct_abstraction(abst_y, y))

        # alpha

        node = helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            alpha=0.5
        )
        a = np.random.ranf([3, 5]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, alpha=0.5)

        _a = Abstraction().load(conf12d, 'a', a.shape, 'FLOAT', a)
        _b = Abstraction().load(conf12d, 'b', b.shape, 'FLOAT', b)
        _c = Abstraction().load(conf12d, 'c', c.shape, 'FLOAT', c)
        abst_y, _ = interp.interp_Gemm([_a, _b, _c], node, 'Gemm', 'y')

        self.assertTrue(correct_abstraction(abst_y, y))

        # beta

        node = helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            beta=0.5
        )
        a = np.random.ranf([2, 7]).astype(np.float32)
        b = np.random.ranf([7, 4]).astype(np.float32)
        c = np.random.ranf([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, beta=0.5)

        _a = Abstraction().load(conf12d, 'a', a.shape, 'FLOAT', a)
        _b = Abstraction().load(conf12d, 'b', b.shape, 'FLOAT', b)
        _c = Abstraction().load(conf12d, 'c', c.shape, 'FLOAT', c)
        abst_y, _ = interp.interp_Gemm([_a, _b, _c], node, 'Gemm', 'y')

        self.assertTrue(correct_abstraction(abst_y, y))

        # default_no_bias

        node = helper.make_node(
            'Gemm',
            inputs=['a', 'b'],
            outputs=['y']
        )
        a = np.random.ranf([2, 10]).astype(np.float32)
        b = np.random.ranf([10, 3]).astype(np.float32)
        y = gemm_reference_implementation(a, b)

        _a = Abstraction().load(conf12d, 'a', a.shape, 'FLOAT', a)
        _b = Abstraction().load(conf12d, 'b', b.shape, 'FLOAT', b)
        abst_y, _ = interp.interp_Gemm([_a, _b], node, 'Gemm', 'y')

        self.assertTrue(correct_abstraction(abst_y, y))


        # default_single_elem_vector_bias

        node = helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )
        a = np.random.ranf([3, 7]).astype(np.float32)
        b = np.random.ranf([7, 3]).astype(np.float32)
        c = np.random.ranf([1]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)

        _a = Abstraction().load(conf12d, 'a', a.shape, 'FLOAT', a)
        _b = Abstraction().load(conf12d, 'b', b.shape, 'FLOAT', b)
        _c = Abstraction().load(conf11d, 'c', c.shape, 'FLOAT', c)
        abst_y, _ = interp.interp_Gemm([_a, _b, _c], node, 'Gemm', 'y')

        self.assertTrue(correct_abstraction(abst_y, y))

        # transpose A

        node = helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            transA=1
        )
        a = np.random.ranf([6, 3]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, transA=1)

        _a = Abstraction().load(conf12d, 'a', a.shape, 'FLOAT', a)
        _b = Abstraction().load(conf12d, 'b', b.shape, 'FLOAT', b)
        _c = Abstraction().load(conf12d, 'c', c.shape, 'FLOAT', c)
        abst_y, _ = interp.interp_Gemm([_a, _b, _c], node, 'Gemm', 'y')

        self.assertTrue(correct_abstraction(abst_y, y))

        # transpose B

        node = helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            transB=1
        )
        a = np.random.ranf([3, 6]).astype(np.float32)
        b = np.random.ranf([4, 6]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, transB=1)

        _a = Abstraction().load(conf12d, 'a', a.shape, 'FLOAT', a)
        _b = Abstraction().load(conf12d, 'b', b.shape, 'FLOAT', b)
        _c = Abstraction().load(conf12d, 'c', c.shape, 'FLOAT', c)
        abst_y, _ = interp.interp_Gemm([_a, _b, _c], node, 'Gemm', 'y')

        self.assertTrue(correct_abstraction(abst_y, y))

    def test_nllloss(self):

        interp = Interpreter()
        conf_exact_1d = AbstractionInitConfig(diff=True, from_init=True, stride=[1])
        conf_exact_2d = AbstractionInitConfig(diff=True, from_init=True, stride=[1,1])
        conf_exact_3d = AbstractionInitConfig(diff=True, from_init=True, stride=[1,1,1])
        conf_strd_2 = AbstractionInitConfig(diff=True, from_init=True, stride=[2,2,2])
        conf_strd_2_2d = AbstractionInitConfig(diff=True, from_init=True, stride=[2,2])
        conf_strd_2_1d = AbstractionInitConfig(diff=True, from_init=True, stride=[2])

        # negative log likelihood loss, "none" reduction
        N, C, d1 = 2, 3, 2
        input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
        target = [[2, 1], [0, 2]]

        loss = np.zeros((N, d1))
        for n in range(N):
            for d_1 in range(d1):
                c = target[n][d_1]
                loss[n][d_1] = -input[n][c][d_1]

        abs_input = Abstraction().load(conf_strd_2, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_exact_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target'],
            outputs=['loss'],
            reduction='none'
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        self.assertTrue(correct_abstraction(abst_loss, loss))

        # =======

        abs_input = Abstraction().load(conf_exact_3d, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_exact_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target'],
            outputs=['loss'],
            reduction='none'
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        self.assertTrue(correct_abstraction(abst_loss, loss, tight=True))

        # weighted negative log likelihood loss, sum reduction
        N, C, d1 = 2, 3, 2
        input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
        target = [[2, 1], [0, 2]]
        weight = [0.2, 0.3, 0.1]
        loss = np.zeros((N, d1))
        for n in range(N):
            for d_1 in range(d1):
                c = target[n][d_1]
                loss[n][d_1] = -input[n][c][d_1] * weight[c]

        # loss = np.array(loss)
        loss = np.sum(loss)

        abs_input = Abstraction().load(conf_strd_2, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_exact_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        abs_weight = Abstraction().load(conf_strd_2_1d, 'weight',  np.array(weight).shape, 'FLOAT', np.array(weight))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            reduction='sum'
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target, abs_weight], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        # abst_loss.print()
        # print(loss)
        self.assertTrue(correct_abstraction(abst_loss, loss))

        # =======

        N, C, d1 = 2, 3, 2
        input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
        target = [[2, 1], [0, 2]]
        weight = [0.2, 0.3, 0.1]
        loss = np.zeros((N, d1))
        for n in range(N):
            for d_1 in range(d1):
                c = target[n][d_1]
                loss[n][d_1] = -input[n][c][d_1] * weight[c]

        # loss = np.array(loss)
        loss = np.sum(loss)

        abs_input = Abstraction().load(conf_exact_3d, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_exact_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        abs_weight = Abstraction().load(conf_exact_1d, 'weight',  np.array(weight).shape, 'FLOAT', np.array(weight))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            reduction='sum'
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target, abs_weight], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        # abst_loss.print()
        # print(loss)
        self.assertTrue(correct_abstraction(abst_loss, loss, tight=True))

        # weighted negative log likelihood loss, mean reduction
        N, C, d1 = 2, 3, 2
        input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
        target = [[2, 1], [0, 2]]
        weight = [0.2, 0.3, 0.1]
        loss = np.zeros((N, d1))
        weight_total = 0
        for n in range(N):
            for d_1 in range(d1):
                c = target[n][d_1]
                loss[n][d_1] = -input[n][c][d_1] * weight[c]
                weight_total = weight_total + weight[c]

        loss = np.sum(loss) / weight_total

        abs_input = Abstraction().load(conf_strd_2, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_exact_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        abs_weight = Abstraction().load(conf_strd_2_1d, 'weight',  np.array(weight).shape, 'FLOAT', np.array(weight))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            # reduction='mean'
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target, abs_weight], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        # abst_loss.print()
        # print(loss)
        self.assertTrue(correct_abstraction(abst_loss, loss))

        # =======

        N, C, d1 = 2, 3, 2
        input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
        target = [[2, 1], [0, 2]]
        weight = [0.2, 0.3, 0.1]
        loss = np.zeros((N, d1))
        weight_total = 0
        for n in range(N):
            for d_1 in range(d1):
                c = target[n][d_1]
                loss[n][d_1] = -input[n][c][d_1] * weight[c]
                weight_total = weight_total + weight[c]

        loss = np.sum(loss) / weight_total

        abs_input = Abstraction().load(conf_exact_3d, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_exact_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        abs_weight = Abstraction().load(conf_exact_1d, 'weight',  np.array(weight).shape, 'FLOAT', np.array(weight))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            reduction='mean'
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target, abs_weight], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        # abst_loss.print()
        # print(loss)
        self.assertTrue(correct_abstraction(abst_loss, loss, tight=True))

        # ===== below we test non-exact target =====

        # negative log likelihood loss, "none" reduction
        N, C, d1 = 2, 3, 2
        input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
        target = [[2, 1], [0, 2]]

        loss = np.zeros((N, d1))
        for n in range(N):
            for d_1 in range(d1):
                c = target[n][d_1]
                loss[n][d_1] = -input[n][c][d_1]

        abs_input = Abstraction().load(conf_strd_2, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_strd_2_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target'],
            outputs=['loss'],
            reduction='mean'
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        # abst_loss.print()
        # print(loss)
        self.assertTrue(correct_abstraction(abst_loss, loss))

        # =======

        N, C, d1 = 2, 3, 2
        input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
        target = [[2, 1], [0, 2]]
        weight = [0.2, 0.3, 0.1]
        loss = np.zeros((N, d1))
        weight_total = 0
        for n in range(N):
            for d_1 in range(d1):
                c = target[n][d_1]
                loss[n][d_1] = -input[n][c][d_1] * weight[c]
                weight_total = weight_total + weight[c]

        loss = np.sum(loss) / weight_total

        abs_input = Abstraction().load(conf_strd_2, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_strd_2_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        abs_weight = Abstraction().load(conf_strd_2_1d, 'weight',  np.array(weight).shape, 'FLOAT', np.array(weight))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            reduction='mean',
            ignore_index=100
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target, abs_weight], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        # abst_loss.print()
        # print(loss)
        self.assertTrue(correct_abstraction(abst_loss, loss))


        # =======

        N, C, d1 = 2, 3, 2
        input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
        target = [[2, 1], [0, 2]]
        weight = [0.2, 0.3, 0.1]
        loss = np.zeros((N, d1))
        weight_total = 0
        for n in range(N):
            for d_1 in range(d1):
                c = target[n][d_1]
                if c != 0:
                    loss[n][d_1] = -input[n][c][d_1] * weight[c]
                    weight_total = weight_total + weight[c]

        loss = np.sum(loss) / weight_total

        abs_input = Abstraction().load(conf_strd_2, 'input', np.array(input).shape, 'FLOAT', np.array(input))
        abs_target = Abstraction().load(conf_exact_2d, 'target', np.array(target).shape, 'FLOAT', np.array(target))
        abs_weight = Abstraction().load(conf_strd_2_1d, 'weight',  np.array(weight).shape, 'FLOAT', np.array(weight))
        node = helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            reduction='mean',
            ignore_index=0
        )
        abst_loss, _ = interp.interp_NegativeLogLikelihoodLoss([abs_input, abs_target, abs_weight], node,
                                                               'NegativeLogLikelihoodLoss', 'loss')
        # abst_loss.print()
        # print(loss)
        self.assertTrue(correct_abstraction(abst_loss, loss))

    def test_ScatterElements(self):

        interp = Interpreter()
        conf_exact = AbstractionInitConfig(diff=True, from_init=True, stride=[1,1])
        conf_strd_2 = AbstractionInitConfig(diff=True, from_init=True, stride=[2,2])
        conf_strd_3 = AbstractionInitConfig(diff=True, from_init=True, stride=[3,3])

        data = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        indices = [
            [1, 0, 2],
            [0, 2, 1],
        ]
        updates = [
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
        ]
        output = [
            [2.0, 1.1, 0.0],
            [1.0, 0.0, 2.2],
            [0.0, 2.1, 1.2]
        ]

        data = np.array(data)
        indices = np.array(indices)
        updates = np.array(updates)
        output = np.array(output)

        node = helper.make_node(
            'ScatterElements',
            inputs=['data', 'indices', 'updates'],
            outputs=['y'],
            axis=0,
        )

        a_data = Abstraction().load(conf_strd_2, 'data', data.shape, 'FLOAT', data)
        a_indices = Abstraction().load(conf_strd_2, 'indices', indices.shape, 'FLOAT', indices)
        a_updates = Abstraction().load(conf_strd_2, 'updates', updates.shape, 'FLOAT', updates)
        a_res, _ = interp.interp_ScatterElements([a_data, a_indices, a_updates], node, 'ScatterElements', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, output))


        a_data = Abstraction().load(conf_strd_3, 'data', data.shape, 'FLOAT', data)
        a_indices = Abstraction().load(conf_strd_2, 'indices', indices.shape, 'FLOAT', indices)
        a_updates = Abstraction().load(conf_strd_2, 'updates', updates.shape, 'FLOAT', updates)
        a_res, _ = interp.interp_ScatterElements([a_data, a_indices, a_updates], node, 'ScatterElements', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, output))

        # ============

        data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        indices = [[1, 3]]
        updates = [[1.1, 2.1]]
        axis = 1
        output = [[1.0, 1.1, 3.0, 2.1, 5.0]]

        data = np.array(data)
        indices = np.array(indices)
        updates = np.array(updates)
        output = np.array(output)

        node = helper.make_node(
            'ScatterElements',
            inputs=['data', 'indices', 'updates'],
            outputs=['y'],
            axis=axis,
        )

        a_data = Abstraction().load(conf_strd_2, 'data', data.shape, 'FLOAT', data)
        a_indices = Abstraction().load(conf_strd_2, 'indices', indices.shape, 'FLOAT', indices)
        a_updates = Abstraction().load(conf_strd_2, 'updates', updates.shape, 'FLOAT', updates)
        a_res, _ = interp.interp_ScatterElements([a_data, a_indices, a_updates], node, 'ScatterElements', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, output))


        a_data = Abstraction().load(conf_exact, 'data', data.shape, 'FLOAT', data)
        a_indices = Abstraction().load(conf_exact, 'indices', indices.shape, 'FLOAT', indices)
        a_updates = Abstraction().load(conf_exact, 'updates', updates.shape, 'FLOAT', updates)
        a_res, _ = interp.interp_ScatterElements([a_data, a_indices, a_updates], node, 'ScatterElements', 'res')
        # a_res.print()
        self.assertTrue(correct_abstraction(a_res, output))

    def test_Expand(self):
        interp = Interpreter()
        conf_exact = AbstractionInitConfig(diff=True, from_init=True, stride=[1])

        conf_strd_5 = AbstractionInitConfig(diff=True, from_init=True, stride=5)
        a = np.random.randn(8,9,1)
        b = np.array([5,1,1,6])

        a_a = Abstraction().load(conf_strd_5, 'a', a.shape, 'FLOAT', a)
        a_b = Abstraction().load(conf_exact, 'b', b.shape, 'FLOAT', b)
        a_res, _ = interp.interp_Expand([a_a, a_b], None, 'Expand', 'res')
        self.assertTrue(correct_format(a_res))
        self.assertListEqual(a_res.shape, [5, 8, 9, 6])

        a = np.random.randn(3,1)
        b = np.array([2,1,6])

        a_a = Abstraction().load(conf_strd_5, 'a', a.shape, 'FLOAT', a)
        a_b = Abstraction().load(conf_exact, 'b', b.shape, 'FLOAT', b)
        a_res, _ = interp.interp_Expand([a_a, a_b], None, 'Expand', 'res')
        self.assertTrue(correct_format(a_res))
        self.assertListEqual(a_res.shape, [2, 3, 6])


        a = np.random.randn(3,1)
        b = np.array([3,4])

        a_a = Abstraction().load(conf_strd_5, 'a', a.shape, 'FLOAT', a)
        a_b = Abstraction().load(conf_exact, 'b', b.shape, 'FLOAT', b)
        a_res, _ = interp.interp_Expand([a_a, a_b], None, 'Expand', 'res')
        self.assertTrue(correct_format(a_res))
        self.assertListEqual(a_res.shape, [3, 4])

    def test_GatherND(self):
        interp = Interpreter()
        conf_exact = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        conf_s3 = AbstractionInitConfig(diff=True, from_init=True, stride=3)


        data = np.array([[[0, 1], [10, 12]], [[20, 21], [6, 7]]], dtype=np.float32)
        indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
        expected_output = np.array([[[10, 12]], [[20, 21]]], dtype=np.float32)

        data = np.expand_dims(np.expand_dims(data, 0), 0)
        indices = np.expand_dims(np.expand_dims(indices, 0), 0)
        expected_output = np.expand_dims(np.expand_dims(expected_output, 0), 0)
        data = np.tile(data, reps=(2, 2, 1, 1, 1))
        expected_output = np.tile(expected_output, reps=(2, 2, 1, 1, 1))
        data += np.array([1,2,3,4]).reshape((2, 2, 1, 1, 1))
        expected_output += np.array([1,2,3,4]).reshape((2, 2, 1, 1, 1))
        indices = np.tile(indices, reps=(2, 2, 1, 1, 1))

        node = helper.make_node(
            'GatherND',
            inputs=['data', 'indices'],
            outputs=['output'],
            batch_dims=2
        )

        a_data = Abstraction().load(conf_exact, 'data', data.shape, 'FLOAT', data)
        a_indices = Abstraction().load(conf_exact, 'indices', indices.shape, 'INT', indices)
        a_output, _ = interp.interp_GatherND([a_data, a_indices], node, 'GatherND', 'output')
        # a_output.print()
        self.assertTrue(correct_abstraction(a_output, expected_output, tight=True))

        a_data = Abstraction().load(conf_s3, 'data', data.shape, 'FLOAT', data)
        a_indices = Abstraction().load(conf_exact, 'indices', indices.shape, 'INT', indices)
        a_output, _ = interp.interp_GatherND([a_data, a_indices], node, 'GatherND', 'output')
        # a_output.print()
        self.assertTrue(correct_abstraction(a_output, expected_output, tight=False))

        # =======

        data = np.array([[0, 1], [2, 3]], dtype=np.int32)
        indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
        expected_output = np.array([0, 3], dtype=np.int32)

        node = helper.make_node(
            'GatherND',
            inputs=['data', 'indices'],
            outputs=['output'],
            batch_dims=0
        )

        a_data = Abstraction().load(conf_exact, 'data', data.shape, 'FLOAT', data)
        a_indices = Abstraction().load(conf_exact, 'indices', indices.shape, 'INT', indices)
        a_output, _ = interp.interp_GatherND([a_data, a_indices], node, 'GatherND', 'output')
        # a_output.print()
        self.assertTrue(correct_abstraction(a_output, expected_output, tight=True))

    def test_Range(self):
        interp = Interpreter()
        conf_exact = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        conf_exact_nodiff = AbstractionInitConfig(diff=False, from_init=True, stride=1)
        conf_s3 = AbstractionInitConfig(diff=True, from_init=True, stride=3)

        start = 3
        limit = 9
        delta = 3

        start = np.array(start)
        limit = np.array(limit)
        delta = np.array(delta)

        a_start = Abstraction().load(conf_exact, 'start', start.shape, 'FLOAT', start)
        a_limit = Abstraction().load(conf_exact, 'limit', limit.shape, 'FLOAT', limit)
        a_delta = Abstraction().load(conf_exact, 'delta', delta.shape, 'FLOAT', delta)
        a_output, _ = interp.interp_Range([a_start, a_limit, a_delta], None, 'Range', 'output')
        # a_output.print()
        self.assertTrue(correct_abstraction(a_output, np.array([3., 6.]), tight=True))


        start = 10
        limit = 4
        delta = -2

        start = np.array(start)
        limit = np.array(limit)
        delta = np.array(delta)

        a_start = Abstraction().load(conf_exact, 'start', start.shape, 'FLOAT', start)
        a_limit = Abstraction().load(conf_exact, 'limit', limit.shape, 'FLOAT', limit)
        a_delta = Abstraction().load(conf_exact, 'delta', delta.shape, 'FLOAT', delta)
        a_output, _ = interp.interp_Range([a_start, a_limit, a_delta], None, 'Range', 'output')
        # a_output.print()
        self.assertTrue(correct_abstraction(a_output, np.array([10, 8, 6]), tight=True))


        start = 10
        limit = 4
        delta = -2

        start = np.array(start)
        limit = np.array(limit)
        delta = np.array(delta)

        a_start = Abstraction().load(conf_exact_nodiff, 'start', start.shape, 'FLOAT', start)
        a_limit = Abstraction().load(conf_exact_nodiff, 'limit', limit.shape, 'FLOAT', limit)
        a_delta = Abstraction().load(conf_exact_nodiff, 'delta', delta.shape, 'FLOAT', delta)
        a_delta.lb -= 1.
        a_delta.ub += 1.
        a_output, _ = interp.interp_Range([a_start, a_limit, a_delta], None, 'Range', 'output')
        # a_output.print()
        # self.assertTrue(correct_abstraction(a_output, np.array([10, 8, 6]), tight=True))

    def test_ArgMinMax(self):
        interp = Interpreter()
        conf_exact = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        conf_s2 = AbstractionInitConfig(diff=True, from_init=True, stride=2)
        conf_s3 = AbstractionInitConfig(diff=True, from_init=True, stride=3)

        for keepdims in [0, 1]:
            for mode in ['ArgMin', 'ArgMax']:
                if mode == 'ArgMin':
                    interp_func = Interpreter.interp_ArgMin
                    check_func = argmin_use_numpy
                else:
                    interp_func = Interpreter.interp_ArgMax
                    check_func = argmax_use_numpy


                data = np.array([[2, 1], [3, 10]], dtype=np.float32)
                node = helper.make_node(
                    mode,
                    inputs=['data'],
                    outputs=['result'],
                    keepdims=keepdims)
                # result: [[1], [1]]
                result = check_func(data, keepdims=keepdims)

                a_data = Abstraction().load(conf_exact, 'data', data.shape, 'FLOAT', data)
                a_result, _ = interp_func(interp, [a_data], node, 'ArgMax', 'result')
                # a_result.print()
                self.assertTrue(correct_abstraction(a_result, result, tight=True))

                node = helper.make_node(
                    mode,
                    inputs=['data'],
                    outputs=['result'],
                    keepdims=keepdims,
                    axis=-1)
                result = check_func(data, keepdims=keepdims, axis=-1)
                a_result, _ = interp_func(interp, [a_data], node, 'ArgMax', 'result')
                # a_result.print()
                self.assertTrue(correct_abstraction(a_result, result, tight=True))

                data = np.random.random((10, 10, 10))
                node = helper.make_node(
                    mode,
                    inputs=['data'],
                    outputs=['result'],
                    keepdims=keepdims,
                    axis=1)
                result = check_func(data, keepdims=keepdims, axis=1)

                a_data = Abstraction().load(conf_exact, 'data', data.shape, 'FLOAT', data)
                a_result, _ = interp_func(interp, [a_data], node, 'ArgMax', 'result')
                # a_result.print()
                self.assertTrue(correct_abstraction(a_result, result, tight=True))

                a_data = Abstraction().load(conf_s3, 'data', data.shape, 'FLOAT', data)
                a_result, _ = interp_func(interp, [a_data], node, 'ArgMax', 'result')
                # a_result.print()
                self.assertTrue(correct_abstraction(a_result, result))

                a_data = Abstraction().load(conf_s2, 'data', data.shape, 'FLOAT', data)
                a_result, _ = interp_func(interp, [a_data], node, 'ArgMax', 'result')
                # a_result.print()
                self.assertTrue(correct_abstraction(a_result, result))

def gemm_reference_implementation(A, B, C=None, alpha=1., beta=1., transA=0,
                                  transB=0):  # type: (np.ndarray, np.ndarray, Optional[np.ndarray], float, float, int, int) -> np.ndarray
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y


def gather_nd_impl(data, indices, batch_dims):
    # type: (np.ndarray, np.ndarray, int) -> np.ndarray
    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    #The list of data/indice shape of batch_dims
    batch_dims_shape = []

    #The number of elements in the batch_dims for data/indice array
    batch_dims_size = 1

    # Check the shape of indice and data are identicial for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below

    # Compute shape of output array
    output_shape = batch_dims_shape + list(indices.shape)[batch_dims:-1] if (indices.shape[-1] == data_rank - batch_dims) \
        else batch_dims_shape + list(indices.shape)[batch_dims:-1] + list(data.shape)[batch_dims + indices.shape[-1]:]

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
    reshaped_data = data.reshape((batch_dims_size, ) + data.shape[batch_dims:])

    # gather each scalar value from 'data'
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim,) + gather_index])
    return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)

def argmax_use_numpy(data, axis=0, keepdims=1):  # type: (np.ndarray, int, int) -> (np.ndarray)
    result = np.argmax(data, axis=axis)
    if (keepdims == 1):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)

def argmin_use_numpy(data, axis=0, keepdims=1):  # type: (np.ndarray, int, int) -> (np.ndarray)
    result = np.argmin(data, axis=axis)
    if (keepdims == 1):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)

def pad_impl(data, raw_pads, mode, constant_values=0.0):  # type: ignore

    input_rank = data.ndim
    if input_rank * 2 != raw_pads.size:
        raise Exception('The number of elements in raw_pads should be 2 * data_rank')

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    pad_width = ()
    for i in range(int(raw_pads.size / 2)):
        pad_width += ((raw_pads[i], raw_pads[i + input_rank])),  # type: ignore

    if mode == 'constant':
        y = np.pad(
            data,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )
        return y

    y = np.pad(
        data,
        pad_width=pad_width,
        mode=mode,
    )

    return y

def logsoftmax(x, axis=-1):  # type: (np.ndarray, int) -> np.ndarray
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return (x - x_max) - np.log(s)

if __name__ == '__main__':
    unittest.main()
    # TestAbstraction().test_Clip()