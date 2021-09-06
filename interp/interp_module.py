
import torch
from torch.nn import Module
import onnx
import onnx.shape_inference
import onnx.numpy_helper

from interp.interp_operator import *
from interp.interp_utils import AbstractionInitConfig, fine_grain_parameters, forbid_fine_grain_flow, discrete_types

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
        # edge format:
        # key: vi, value: list
        # where each element of value corresponds to (vj, vi index in node input, vj index in node output, node name, node)
        self.edges = dict()
        self.all_nodes = set()

        for node in self.onnx_model.graph.node:
            if node.domain != '':
                print(f"  found domain def: {node.domain}")

            for v in list(node.input) + list(node.output):
                if v not in self.signature_dict:
                    print(f'  warning: {v} has not appeared yet')
                self.all_nodes.add(v)

            for i, vi in enumerate(list(node.input)):
                for j, vj in enumerate(list(node.output)):
                    if vi not in self.edges: self.edges[vi] = list()
                    self.edges[vi].append((vj, i, j, node.op_type, node.name, node))

                    if vi not in self.deg_out:
                        self.deg_out[vi] = 1
                    else:
                        self.deg_out[vi] += 1
                    if vj not in self.deg_in:
                        self.deg_in[vj] = 1
                    else:
                        self.deg_in[vj] += 1

        for v in self.all_nodes:
            if v not in self.deg_in: self.deg_in[v] = 0
            if v not in self.deg_out: self.deg_out[v] = 0
            if v not in self.edges: self.edges[v] = list()

        self.start_points = set([x for x in self.all_nodes if self.deg_in[x] == 0])

        """construct node dictionary"""
        self.node_dict = dict([(x.name, x) for x in self.onnx_model.graph.node])
        self.node_types = set([x.op_type for x in self.node_dict.values()])

        """summary"""
        print('==== Model Summary ====')
        print('Number of nodes:', len(self.signature_dict))
        print('Number of edges:', sum([len(x) for x in self.edges.values()]))
        print('Number of start points:', len(self.start_points))
        if len(self.start_points) <= 5:
            print('  They are', self.start_points)
        print('Number of Op types:', len(self.node_types))
        # if len(self.node_types) <= 5:
        print('  They are', self.node_types)
        print('=======================')

        """Space for analysis"""
        self.initial_abstracts = None
        self.abstracts = None
        # key: var_name, value: tuple of (list of numerical error exceptions, set of caused tensors)
        self.possible_numerical_errors = dict()


    def _shape_invertor(self, shape):
        """
            Converty shape in TensorShapeProto format to list of dims
        :param shape:
        :return:
        """
        assert isinstance(shape, onnx.TensorShapeProto)
        ans = [now_dim.dim_param if now_dim.dim_param != '' else
               (None if now_dim.dim_value is None else now_dim.dim_value) for now_dim in shape.dim]
        return ans

    def analyze(self, init_config=None):

        # independent abstraction variables
        self.initial_abstracts = dict()

        if init_config is None:
            init_config = dict()

        print('find variable for fine grain abstraction...', flush=True)
        # this part requires topology sort in the reversed graph
        # byproduct: reverse edges
        rev_edges = dict()
        # byproduct: op types involved for given variable
        involved_op_types = dict()
        for k, v in self.edges.items():
            for item in v:
                vj = item[0]
                if vj not in rev_edges: rev_edges[vj] = list()
                rev_edges[vj].append((k,) + item[1:])
        cur_deg_in = self.deg_out.copy()
        queue = [k for k,v in cur_deg_in.items() if v == 0]
        require_fine_grain_vars = set()
        l = 0
        while l < len(queue):
            cur_var = queue[l]
            if cur_var in rev_edges:
                for vi, ind_i, ind_j, node_optype, node_name, node in rev_edges[cur_var]:
                    cur_deg_in[vi] -= 1
                    if vi not in involved_op_types:
                        involved_op_types[vi] = set()
                    involved_op_types[vi].add(node_optype)
                    if cur_var in require_fine_grain_vars and not node_optype in forbid_fine_grain_flow:
                        require_fine_grain_vars.add(vi)
                    if node_optype in fine_grain_parameters and ind_i in fine_grain_parameters[node_optype]:
                        require_fine_grain_vars.add(vi)
                    if cur_deg_in[vi] == 0:
                        queue.append(vi)
            l += 1
        print('  ', len(require_fine_grain_vars), 'fine grain variables found')

        print('initialize abstractions...', flush=True)
        fine_grain_config = AbstractionInitConfig(diff=True, stride=1, from_init=True)
        for s in self.start_points:
            if s not in init_config:
                if s in require_fine_grain_vars:
                    init_config[s] = fine_grain_config
                    init_config[s].diff = self.signature_dict[s][0] not in discrete_types
                else:
                    if s in self.input_vars:
                        # for input tensors, usually we don't use raw initialized data to load the Abstraction
                        init_config[s] = AbstractionInitConfig(diff=self.signature_dict[s][0] not in discrete_types, stride=-1, from_init=False)
                    else:
                        # for weights, to get a better sense of the weight range, we use raw initialized data to load
                        init_config[s] = AbstractionInitConfig(diff=self.signature_dict[s][0] not in discrete_types, stride=-1, from_init=True)
            else:
                assert isinstance(init_config[s], AbstractionInitConfig)

            now_t, now_shape = self.signature_dict[s]
            now_raw_data = self.initializer_dict[s][1] if s in self.initializer_dict else None
            self.initial_abstracts[s] = Abstraction()
            self.initial_abstracts[s].load(init_config[s], s, now_shape, now_t, now_raw_data)

        # whole abstract variables
        self.abstracts = dict()
        for k, v in self.initial_abstracts.items():
            self.abstracts[k] = v

        interpreter = Interpreter()

        print('topology sort based intepretation...', flush=True)
        queue = list(self.start_points).copy()
        cur_deg_in = self.deg_in.copy()
        l = 0
        while l < len(queue):
            cur_var = queue[l]
            for vj, ind_i, ind_j, node_optype, node_name, node in self.edges[cur_var]:
                cur_deg_in[vj] -= 1
                if cur_deg_in[vj] == 0:
                    if vj in self.abstracts:
                        # already obtained the abstraction from previous runs,
                        # only happens for nodes with multiple outputs,
                        # so now we can skip
                        queue.append(vj)
                        pass

                    else:
                        cur_abst, cur_exceps = interpreter.handle(
                            [self.abstracts[x] for x in node.input], node, node_optype, vj
                        )
                        if len(cur_exceps) > 0:
                            roots = list()
                            for x in node.input:
                                if x in self.possible_numerical_errors:
                                    roots.extend(self.possible_numerical_errors[x][1])
                            roots = set(roots)
                            if len(roots) == 0:
                                roots = {node}
                            self.possible_numerical_errors[vj] = (cur_exceps, roots)
                        if cur_abst is None:
                            print(f'! No abstraction generated for {vj}: '
                                  f'node name = {node_name}, type = {node_optype}')
                        else:
                            queue.append(vj)
                            if isinstance(cur_abst, Abstraction):
                                # single output node
                                self.abstracts[vj] = cur_abst
                            else:
                                # multiple output node: execute once, update all output nodes
                                for i, cur_cur_abst in enumerate(cur_abst):
                                    self.abstracts[node.output[i]] = cur_cur_abst
            l += 1

        return self.possible_numerical_errors


def load_onnx_from_file(path):
    onnx_model = onnx.load_model(path)
    return InterpModule(onnx_model)

