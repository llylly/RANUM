import unittest
import torch
import numpy as np

import onnx
from onnx import helper
from onnx.helper import make_tensor_value_info, make_sequence_value_info
from functools import reduce

from interp.interp_utils import AbstractionInitConfig
from interp.interp_operator import Abstraction, Interpreter

from tests.test_abstraction import tf_equal, correct_abstraction


class TestAbstraction(unittest.TestCase):

    def abst_shape_check(self, obj: Abstraction):
        target_size = [len(x) for x in obj.splits]
        self.assertEqual(obj.get_dim(), len(obj.splits))
        self.assertEqual(obj.lb.dim(), obj.get_dim())
        self.assertEqual(obj.ub.dim(), obj.get_dim())
        self.assertEqual(obj.lb.shape, torch.Size(target_size))
        self.assertEqual(obj.ub.shape, torch.Size(target_size))

    def test_loop_11(self):
        # case loop_11 from https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop

        # Given a tensor x of values [x1, ..., xN], and initial tensor y
        # sum up its elements using a scan
        # returning the final state (y+x1+x2+...+xN) as well the scan_output
        # [y+x1, y+x1+x2, ..., y+x1+x2+...+xN]

        y_in = onnx.helper.make_tensor_value_info('y_in', onnx.TensorProto.FLOAT, [1])
        y_out = onnx.helper.make_tensor_value_info('y_out', onnx.TensorProto.FLOAT, [1])
        scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [1])
        cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
        cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
        iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([-2]).astype(np.float32)

        x_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['x'],
            value=onnx.helper.make_tensor(
                name='const_tensor_x',
                data_type=onnx.TensorProto.FLOAT,
                dims=x.shape,
                vals=x.flatten().astype(float),
            )
        )

        one_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one'],
            value=onnx.helper.make_tensor(
                name='const_tensor_one',
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[1]
            )
        )

        i_add_node = onnx.helper.make_node(
            'Add',
            inputs=['iter_count', 'one'],
            outputs=['end']
        )

        start_unsqueeze_node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['iter_count'],
            outputs=['slice_start'],
            axes=[0]
        )

        end_unsqueeze_node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['end'],
            outputs=['slice_end'],
            axes=[0]
        )

        slice_node = onnx.helper.make_node(
            'Slice',
            inputs=['x', 'slice_start', 'slice_end'],
            outputs=['slice_out']
        )

        y_add_node = onnx.helper.make_node(
            'Add',
            inputs=['y_in', 'slice_out'],
            outputs=['y_out']
        )

        identity_node = onnx.helper.make_node(
            'Identity',
            inputs=['cond_in'],
            outputs=['cond_out']
        )

        scan_identity_node = onnx.helper.make_node(
            'Identity',
            inputs=['y_out'],
            outputs=['scan_out']
        )

        loop_body = onnx.helper.make_graph(
            [identity_node, x_const_node, one_const_node, i_add_node,
             start_unsqueeze_node, end_unsqueeze_node, slice_node, y_add_node,
             scan_identity_node],
            'loop_body',
            [iter_count, cond_in, y_in],
            [cond_out, y_out, scan_out]
        )

        node = onnx.helper.make_node(
            'Loop',
            inputs=['trip_count', 'cond', 'y'],
            outputs=['res_y', 'res_scan'],
            body=loop_body
        )

        trip_count = np.array(5).astype(np.int64)
        res_y = np.array([13]).astype(np.float32)
        cond = np.array(1).astype(bool)
        res_scan = np.array([-1, 1, 4, 8, 13]).astype(np.float32).reshape((5, 1))
        # expect(node, inputs=[trip_count, cond, y], outputs=[res_y, res_scan],
        #        name='test_loop11', opset_imports=[onnx.helper.make_opsetid("", 11)])

        interp = Interpreter()
        cfg_precise = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        abst_trip_count = Abstraction().load(cfg_precise, 'trip_count', trip_count.shape, 'INT', trip_count)
        abst_cond = Abstraction().load(cfg_precise, 'cond', cond.shape, 'BOOL', cond)
        abst_y = Abstraction().load(cfg_precise, 'y', y.shape, 'INT', y)

        outputs, _ = interp.interp_Loop([abst_trip_count, abst_cond, abst_y], node, 'Loop', 'res_y')

        self.assertTrue(correct_abstraction(outputs[0], np.array([13]), tight=True))
        self.assertTrue(correct_abstraction(outputs[1], np.array([-1,1,4,8,13]).reshape((5,1)), tight=True))

    def test_loop_13(self):
        # case loop_13 from https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop
        # Given a tensor x of values [x1, ..., xN],
        # Return a sequence of tensors of
        #   [[x1], [x1, x2], ..., [x1, ..., xN]]

        seq_in = onnx.helper.make_sequence_value_info('seq_in', onnx.TensorProto.FLOAT, None)
        seq_out = onnx.helper.make_sequence_value_info('seq_out', onnx.TensorProto.FLOAT, None)
        cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
        cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
        iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        x_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['x'],
            value=onnx.helper.make_tensor(
                name='const_tensor_x',
                data_type=onnx.TensorProto.FLOAT,
                dims=x.shape,
                vals=x.flatten().astype(float),
            )
        )

        one_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one'],
            value=onnx.helper.make_tensor(
                name='const_tensor_one',
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[1]
            )
        )

        zero_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['slice_start'],
            value=onnx.helper.make_tensor(
                name='const_tensor_zero',
                data_type=onnx.TensorProto.INT64,
                dims=(1,),
                vals=[0]
            )
        )

        axes_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['axes'],
            value=onnx.helper.make_tensor(
                name='const_tensor_axes',
                data_type=onnx.TensorProto.INT64,
                dims=(1,),
                vals=[0]
            )
        )

        add_node = onnx.helper.make_node(
            'Add',
            inputs=['iter_count', 'one'],
            outputs=['end']
        )

        # end_unsqueeze_node = onnx.helper.make_node(
        #     'Unsqueeze',
        #     inputs=['end', 'axes'],
        #     outputs=['slice_end']
        # )

        slice_node = onnx.helper.make_node(
            'Slice',
            inputs=['x', 'slice_start', 'end'],
            outputs=['slice_out']
        )

        insert_node = onnx.helper.make_node(
            'SequenceInsert',
            inputs=['seq_in', 'slice_out'],
            outputs=['seq_out']
        )

        identity_node = onnx.helper.make_node(
            'Identity',
            inputs=['cond_in'],
            outputs=['cond_out']
        )

        loop_body = onnx.helper.make_graph(
            [identity_node, x_const_node, one_const_node, zero_const_node, add_node,
             axes_node, slice_node, insert_node],
            'loop_body',
            [iter_count, cond_in, seq_in],
            [cond_out, seq_out]
        )

        node = onnx.helper.make_node(
            'Loop',
            inputs=['trip_count', 'cond', 'seq_empty'],
            outputs=['seq_res'],
            body=loop_body
        )

        trip_count = np.array(5).astype(np.int64)
        seq_empty = []  # type: List[Any]
        seq_res = [x[:int(i)] for i in x]
        cond = np.array(1).astype(bool)


        interp = Interpreter()
        cfg_precise = AbstractionInitConfig(diff=True, from_init=True, stride=1)
        abst_trip_count = Abstraction().load(cfg_precise, 'trip_count', trip_count.shape, 'INT', trip_count)
        abst_seq_empty = Abstraction().load(cfg_precise, 'seq_empty', [1], 'INT', seq_empty)
        abst_cond = Abstraction().load(cfg_precise, 'cond', cond.shape, 'BOOL', cond)

        outputs, _ = interp.interp_Loop([abst_trip_count, abst_cond, abst_seq_empty], node, 'Loop', 'output')

        seq_res = [x[:int(i)] for i in x]
        # outputs[0].print()

        self.assertTrue(correct_abstraction(outputs[0], seq_res))



if __name__ == '__main__':
    unittest.main()