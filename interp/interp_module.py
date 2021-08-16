
import torch
from torch.nn import Module
import onnx
import onnx.shape_inference
import onnx.numpy_helper

from interp.interp_operator import *

class InterpModule():

    def __init__(self, onnx_model, debug=True):
        self.onnx_model = onnx_model

        """init type mappers"""
        self.tensor_type_mapper = dict([(id, x.name) for id, x in enumerate(onnx.TensorProto.DataType._enum_type.values)])
        self.attri_type_mapper = dict([(id, name) for id, name in enumerate(onnx.AttributeProto.AttributeType.keys())])

        if len(self.onnx_model.graph.value_info) == 0:
            if debug: print('run model checking and shape inference', flush=True)
            onnx.checker.check_model(self.onnx_model)
            self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
            # even after shape inference, they is still unknown dimensions in shape
            # therefore, when conducting interpretation, we should also fill in the unknown shape elements at the same time

        """retrieve signature dictionary"""
        if debug: print('retrieve signatures from value_info', flush=True)
        # key: name, value: (type in str, shape in list)
        self.signature_dict = dict([(node.name,
                                        (self.tensor_type_mapper[node.type.tensor_type.elem_type],
                                         self._shape_invertor(
                                             node.type.tensor_type.shape
                                         ))) for node in self.onnx_model.graph.value_info])

        """collect initialized vars and init vars"""
        if debug: print('retrieve initializers', flush=True)
        self.initializer_vars = set([node.name for node in self.onnx_model.graph.initializer])
        # key: name, value: (type in str, raw_data in numpy)
        self.initializer_dict = dict([(node.name,
                                       (self.tensor_type_mapper[node.data_type], onnx.numpy_helper.to_array(node))
                                            ) for node in self.onnx_model.graph.initializer])
        for k, v in self.initializer_dict.items():
            # extract the type signatures from graph initializer
            if k not in self.signature_dict:
                self.signature_dict[k] = (v[0], list(v[1].shape))


        """collect input vars"""
        if debug: print('retrieve input vars', flush=True)
        self.input_vars = set([node.name for node in self.onnx_model.graph.input])
        # update the signature dict
        for node in self.onnx_model.graph.input:
            if node.name not in self.signature_dict:
                if debug: print(f'  add input var {node.name} into the signature dict')
                self.signature_dict[node.name] = (self.tensor_type_mapper[node.type.tensor_type.elem_type],
                                                  self._shape_invertor(node.type.tensor_type.shape))

        """collect output vars"""
        if debug: print('retrieve output vars', flush=True)
        self.output_vars = set([node.name for node in self.onnx_model.graph.output])
        # update the signature dict
        for node in self.onnx_model.graph.output:
            if node.name not in self.signature_dict:
                if debug: print(f'  add output var {node.name} into the signature dict')
                self.signature_dict[node.name] = (self.tensor_type_mapper[node.type.tensor_type.elem_type],
                                                  self._shape_invertor(node.type.tensor_type.shape))

        """collect shape identifiers and set them to 1 (corresponding to batch size 1) """
        if debug: print('retriever shape identifiers', flush=True)
        self.shape_identifiers = set()
        for values in self.signature_dict.values():
            for s in values[1]:
                if isinstance(s, str):
                    self.shape_identifiers.add(s)
        print(f'  shape identifiers are {self.shape_identifiers}, all are set to 1 (batch size=1 case)')

        for values in self.signature_dict.values():
            for id, item in enumerate(values[1]):
                if isinstance(item, str):
                    values[1][id] = 1

        """construct the computational graph, and check whether there are unknown variables"""
        if debug: print('construct graph')
        self.deg_in = dict()
        self.deg_out = dict()
        self.edges = dict()
        self.all_nodes = set()

        for node in self.onnx_model.graph.node:
            if node.domain != '':
                print(f"  found domain def: {node.domain}")
            for v in node.input:
                if v not in self.deg_out:
                    self.deg_out[v] = 1
                else:
                    self.deg_out[v] += 1
            for v in node.output:
                if v not in self.deg_in:
                    self.deg_in[v] = 1
                else:
                    self.deg_in[v] += 1

            for v in list(node.input) + list(node.output):
                if v not in self.signature_dict:
                    print(f'  warning: {v} has not appeared yet')
                self.all_nodes.add(v)

            for i, vi in enumerate(list(node.input)):
                for j, vj in enumerate(list(node.output)):
                    print(vi, vj)
                    if vi not in self.edges: self.edges[vi] = list()
                    self.edges[vi].append((vj, i, j, node.op_type, node.name, node))

        for v in self.all_nodes:
            if v not in self.deg_in: self.deg_in[v] = 0
            if v not in self.deg_out: self.deg_out[v] = 0
            if v not in self.edges: self.edges[v] = list()

        self.start_points = set([x for x in self.all_nodes if self.deg_in[x] == 0])

        """summary"""
        print('==== Model Summary ====')
        print('Number of nodes:', len(self.signature_dict))
        print('Number of edges:', sum([len(x) for x in self.edges.values()]))
        print('Number of start points:', len(self.start_points))
        if len(self.start_points) <= 5:
            print('  They are', self.start_points)

    def _shape_invertor(self, shape):
        assert isinstance(shape, onnx.TensorShapeProto)
        ans = [now_dim.dim_param if now_dim.dim_param != '' else
               (None if now_dim.dim_value is None else now_dim.dim_value) for now_dim in shape.dim]
        return ans


def load_onnx_from_file(path):
    onnx_model = onnx.load_model(path)
    return InterpModule(onnx_model)


def init_input_vars():
    # TODO
    pass


def analyze(module: InterpModule):
    # TODO
    for s in module.start_points:
        pass