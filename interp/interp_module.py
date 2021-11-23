import torch
from torch.nn import Module
import onnx
import onnx.shape_inference
import onnx.numpy_helper

from interp.interp_operator import *
from interp.interp_utils import AbstractionInitConfig, fine_grain_parameters, forbid_fine_grain_flow, discrete_types, \
    PossibleNumericalError
from interp.specified_ranges import SpecifiedRanges


class InterpModule():
    SUPER_START_NODE = "SUPER_START"

    def __init__(self, onnx_model, debug=True, customize_shape=None):
        self.onnx_model = onnx_model

        """init type mappers"""
        self.tensor_type_mapper = dict(
            [(id, x.name) for id, x in enumerate(onnx.TensorProto.DataType._enum_type.values)])
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

        """collect constant operators as initializers"""
        if debug: print('retrieve constant operators', flush=True)
        for node in self.onnx_model.graph.node:
            if node.op_type == 'Constant':
                attr = parse_attribute(node)
                if 'value' in attr:
                    value = attr['value']
                    dtype = self.tensor_type_mapper[value.data_type]
                    data = onnx.numpy_helper.to_array(value)
                    # print(dtype, data, node.output[0])
                    self.initializer_vars.add(node.output[0])
                    self.initializer_dict[node.output[0]] = (dtype, data)
                else:
                    raise Exception(
                        f'I encountered an unsupported value field: {list(attr.keys())}, need implementation')

        """collect initializers' signatures"""
        if debug: print('retrieve signatures of initializers', flush=True)
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
        print(
            f'  shape identifiers are {self.shape_identifiers}, all are set to 1 (batch size=1 case) except specified')

        for values in self.signature_dict.values():
            for id, item in enumerate(values[1]):
                if isinstance(item, str):
                    values[1][id] = 1 if customize_shape is None or item not in customize_shape else customize_shape[
                        item]

        """construct the computational graph, and check whether there are unknown variables"""
        if debug: print('construct graph')
        self.deg_in = dict()
        self.deg_out = dict()
        # edge format:
        # key: vi, value: list
        # where each element of value corresponds to (vj, vi index in node input, vj index in node output, node name, node)
        self.edges = dict()
        self.all_nodes = set()
        self.nodes_to_check = set()
        self.start_points = set()

        for node in self.onnx_model.graph.node:
            if node.domain != '':
                print(f"  found domain def: {node.domain}")

            if node.op_type in PossibleNumericalError.OPs2Check:
                self.nodes_to_check.update(list(node.output))

            for v in list(node.input) + list(node.output):
                if v not in self.signature_dict:
                    print(f'  warning: {v} has not appeared yet')
                self.all_nodes.add(v)

            node_input = list(node.input)
            if len(node_input) == 0 and node.op_type in ['RandomNormal', 'RandomUniform']:
                node_input.append(InterpModule.SUPER_START_NODE)
                self.start_points.add(InterpModule.SUPER_START_NODE)

            for i, vi in enumerate(node_input):
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

        """inspect the subgraph of loop, and establish the dependency for unreferenced loop input"""
        self.loop_dependencies = dict()
        # These unreferenced loop inputs should be illegal and should not appear in the model
        # However, I find tf2onnx does construct models with such troublesome things, so we need to process them
        for node in self.onnx_model.graph.node:
            if node.op_type == 'Loop':
                self.loop_dependencies[node.name] = set()
                subgraph = onnx.helper.get_attribute_value(node.attribute[0])
                inner_inputs = set()
                legal_nodes = set().union([inp.name for inp in subgraph.input])
                for inner_node in subgraph.node:
                    legal_nodes = legal_nodes.union(inner_node.output)
                    inner_inputs = inner_inputs.union(inner_node.input)
                illegal_inputs = [x for x in inner_inputs if x not in legal_nodes]
                if len(illegal_inputs) != 0:
                    print(f'  deployed a fix for illegel input detected @ loop {node.name}:', illegal_inputs)
                    self.loop_dependencies[node.name] = self.loop_dependencies[node.name].union(illegal_inputs)

                    for v in list(illegal_inputs) + list(node.output):
                        if v not in self.signature_dict:
                            print(f'  warning: {v} has not appeared yet')
                        self.all_nodes.add(v)

                    for i, vi in enumerate(list(illegal_inputs)):
                        for j, vj in enumerate(list(node.output)):
                            if vi not in self.edges: self.edges[vi] = list()
                            self.edges[vi].append((vj, -1, j, node.op_type, node.name, node))

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

        self.start_points.update([x for x in self.all_nodes if self.deg_in[x] == 0])

        """construct node dictionary"""
        self.node_dict = dict([(x.name, x) for x in self.onnx_model.graph.node])
        self.node_types = set([x.op_type for x in self.node_dict.values()])
        self.unimplemented_types = [x for x in self.node_types if
                                    'interp_' + x not in Interpreter.__dir__(Interpreter())]

        """summary"""
        print('==== Model Summary ====')
        print('Number of nodes:', len(self.signature_dict))
        print('Number of edges:', sum([len(x) for x in self.edges.values()]))
        print('Number of start points:', len(self.start_points))
        # if len(self.start_points) <= 5:
        print('  They are', self.start_points)
        print('Number of op types:', len(self.node_types))
        # if len(self.node_types) <= 5:
        print('  They are', self.node_types)
        print(f'  {len(self.unimplemented_types)} not implemented:', self.unimplemented_types)
        print('=======================')

        """Space for analysis"""
        self.initial_abstracts = None
        self.abstracts = None
        self.require_fine_grain_vars = None
        # key: var_name, value: tuple of (list of numerical error exceptions, set of caused tensors)
        self.possible_numerical_errors = dict()

        """Space for reusable queue"""
        self.queue = None

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

    def analyze(self, init_config=None, interp_config=dict(),
                input_config=AbstractionInitConfig(diff=True, stride=-1, from_init=False, from_init_margin=1.),
                weight_config=AbstractionInitConfig(diff=True, stride=-1, from_init=True, from_init_margin=1.),
                terminate=None):
        """

        :param init_config:
        :param interp_config:
        :param input_config:
        :param weight_config:
        :param terminate: terminate condition, if none, then traverse all involved_vars;
        else it should be a str, denoting a termination after the abst of the node with specified name is generated
        :return:
        """

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
        queue = [k for k, v in cur_deg_in.items() if v == 0]
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
        print(f'    They are {require_fine_grain_vars}')
        self.require_fine_grain_vars = require_fine_grain_vars

        # compute the involved nodes with respected to the op2check
        queue = list(self.nodes_to_check)
        l = 0
        involved_vars = set(queue)
        while l < len(queue):
            cur_var = queue[l]
            if cur_var in rev_edges:
                for vi, ind_i, ind_j, node_optype, node_name, node in rev_edges[cur_var]:
                    if vi not in involved_vars:
                        queue.append(vi)
                        involved_vars.add(vi)
            l += 1
        print(f"number of involved vars: {len(involved_vars)}")

        print('initialize abstractions...', flush=True)
        fine_grain_config = AbstractionInitConfig(diff=True, stride=1, from_init=True)
        specified_inputs = set(init_config.keys())
        for s in self.start_points:
            if s == InterpModule.SUPER_START_NODE: continue
            if s not in init_config:
                if s.lower().count('moving_variance') > 0:
                    # looks like a variance
                    print(
                        f'Parameter {s} looks like a variance, abstract by {AbstractionInitConfig.VARIANCE_CONFIG_DEFAULT}')
                    init_config[s] = AbstractionInitConfig(diff=False, from_init=True,
                                                           lb=AbstractionInitConfig.VARIANCE_CONFIG_DEFAULT[0],
                                                           ub=AbstractionInitConfig.VARIANCE_CONFIG_DEFAULT[1])
                    init_config[s].diff = self.signature_dict[s][0] not in discrete_types
                    specified_inputs.add(s)
                elif s.lower().count('local_step') > 0:
                    # looks like a step
                    print(
                        f'Parameter {s} looks like a step, abstract by [1, inf]')
                    init_config[s] = AbstractionInitConfig(diff=False, from_init=True,
                                                           lb=1,
                                                           ub=PossibleNumericalError.OVERFLOW_LIMIT)
                    init_config[s].diff = self.signature_dict[s][0] not in discrete_types
                    specified_inputs.add(s)
                elif s in require_fine_grain_vars:
                    init_config[s] = fine_grain_config
                    init_config[s].diff = self.signature_dict[s][0] not in discrete_types
                else:
                    if s in self.input_vars:
                        # for input tensors, usually we don't use raw initialized data to load the Abstraction
                        now_config = AbstractionInitConfig(False).load_from_dict(input_config.to_dict())
                        now_config.diff = not self.signature_dict[s][0] in discrete_types
                        init_config[s] = now_config
                    else:
                        # for weights, to get a better sense of the weight range, we use raw initialized data to load
                        now_config = AbstractionInitConfig(False).load_from_dict(weight_config.to_dict())
                        now_config.diff = not self.signature_dict[s][0] in discrete_types
                        init_config[s] = now_config
            else:
                print(f"Parameter {s} is in init_config.")
                assert isinstance(init_config[s], AbstractionInitConfig)
                if s in require_fine_grain_vars:
                    init_config[s].stride = 1

            now_t, now_shape = self.signature_dict[s]
            now_raw_data = self.initializer_dict[s][
                1] if s in self.initializer_dict and s not in specified_inputs else None
            self.initial_abstracts[s] = Abstraction()
            self.initial_abstracts[s].load(init_config[s], s, now_shape, now_t, now_raw_data)

        # whole abstract variables
        self.abstracts = dict()
        for k, v in self.initial_abstracts.items():
            self.abstracts[k] = v

        interpreter = Interpreter(**interp_config)

        print('topology sort based interpretation...', flush=True)
        queue = list(self.start_points).copy()
        """self.queue is a list of tuple (node_name, node context)"""
        """ where node context is 
        (previous node name, 
        node_name, 
        index of previous node name in inputs, 
        index of current node_name in outputs, 
        node optype, 
        node name, 
        node object) """
        self.queue = [(item, None) for item in queue]
        cur_deg_in = self.deg_in.copy()
        l = 0
        # try:
        analyze_result_pos = {}
        analyze_result_neg = {}
        print('=' * 10, "Details", '=' * 10)
        optimize_op_filter = {"Mul",  # -> Square
                              "Sub",  # -> Zero
                              "Slice", "Concat", "Resize", "Sum"}
        while l < len(queue):
            cur_var = queue[l]
            l += 1
            if terminate is None:
                if cur_var not in involved_vars:
                    continue
            else:
                if terminate in self.abstracts:
                    break
            for vj, ind_i, ind_j, node_optype, node_name, node in self.edges[cur_var]:
                cur_deg_in[vj] -= 1
                if cur_deg_in[vj] == 0:
                    if vj in self.abstracts:
                        # already obtained the abstraction from previous runs,
                        # only happens for nodes with multiple outputs,
                        # so now we can skip
                        queue.append(vj)
                        self.queue.append((vj, (cur_var, vj, ind_i, ind_j, node_optype, node_name, node)))
                        pass

                    else:
                        node_input_cnt = len(node.input)
                        unique_node_input_cnt = len(set([x for x in node.input]))
                        # detect potential optimizing operators
                        if unique_node_input_cnt != node_input_cnt and node.op_type not in optimize_op_filter:
                            print(f"({node.name}, {node.op_type}) can be optimized!\n{node.input}")
                        if node.op_type != 'Loop':
                            # optimize multiplication with two same inputs as square
                            if node.op_type == "Mul" and unique_node_input_cnt == 1:
                                cur_abst, cur_exceps = interpreter.handle(
                                    [self.abstracts[node.input[0]]], None, "Square", vj
                                )
                            else:
                                cur_abst, cur_exceps = interpreter.handle(
                                    [self.abstracts[x] for x in node.input], node, node_optype, vj
                                )
                        else:
                            # specially handle the loop dependencies
                            cur_abst, cur_exceps = interpreter.interp_Loop(
                                [self.abstracts[x] for x in node.input], node, node_optype, vj,
                                loop_dependencies=dict(
                                    [(x, self.abstracts[x]) for x in self.loop_dependencies[node.name] if
                                     x in self.abstracts])
                            )
                        if len(cur_exceps) > 0:
                            roots = list()
                            for x in node.input:
                                if x in self.possible_numerical_errors:
                                    roots.extend(self.possible_numerical_errors[x][1])
                            roots = set(roots)
                            if len(roots) == 0:
                                roots = {node.name}
                                catastro = False
                            else:
                                catastro = True
                            self.possible_numerical_errors[vj] = (cur_exceps, roots, catastro)
                            print(f'node name = {node_name}, type = {node_optype} can cause numerical error!')
                            analyze_result_pos[node.op_type] = analyze_result_pos.get(node.op_type, 0) + 1
                        elif node.op_type in PossibleNumericalError.OPs2Check:
                            print(f'node name = {node_name}, type = {node_optype} is safe.')
                            analyze_result_neg[node.op_type] = analyze_result_neg.get(node.op_type, 0) + 1
                        if cur_abst is None:
                            # still enqueue to detect the error with rerunning
                            self.queue.append((vj, (cur_var, vj, ind_i, ind_j, node_optype, node_name, node)))
                            print(f'! No abstraction generated for {vj}: '
                                  f'node name = {node_name}, type = {node_optype}')
                        else:
                            queue.append(vj)
                            self.queue.append((vj, (cur_var, vj, ind_i, ind_j, node_optype, node_name, node)))
                            if isinstance(cur_abst, Abstraction):
                                if PossibleNumericalError.is_invalid(cur_abst.lb) or PossibleNumericalError.is_invalid(
                                        cur_abst.ub):
                                    warnings.warn(f'! The abstraction generated for {vj} is invalid: '
                                                  f'node name = {node_name}, type = {node_optype}')
                                if any(x != len(y) for x, y in zip(cur_abst.lb.shape, cur_abst.splits)):
                                    raise AssertionError(f'! The splits for {vj} does not match the lb(ub).shape: '
                                                         f'node name = {node_name}, type = {node_optype}')
                                # single output node
                                self.abstracts[vj] = cur_abst
                            else:
                                # multiple output node: execute once, update all output nodes
                                for i, cur_cur_abst in enumerate(cur_abst):
                                    if cur_cur_abst is None:
                                        # print(f'! No abstraction generated for the {i}-th output of {vj}: '
                                        #       f'node name = {node_name}, type = {node_optype}')
                                        continue
                                    if PossibleNumericalError.is_invalid(
                                            cur_cur_abst.lb) or PossibleNumericalError.is_invalid(
                                        cur_cur_abst.ub):
                                        warnings.warn(f'! The {i}-th abstraction generated for {vj} is invalid: '
                                                      f'node name = {node_name}, type = {node_optype}')
                                    if any(x != len(y) for x, y in zip(cur_cur_abst.lb.shape, cur_cur_abst.splits)):
                                        raise AssertionError(f'! The splits for {vj} does not match the lb(ub).shape: '
                                                             f'node name = {node_name}, type = {node_optype}')
                                    # if node.output[i] in self.abstracts:
                                    #     print(f'!!!!!! multiple updates on node {node.output[i]} which should not happen')
                                    self.abstracts[node.output[i]] = cur_cur_abst
        # except:
        #     print('error countered')

        # place to inspect abstraction for debug
        # self.abstracts['dropout/truediv/x:0'].print()
        # self.abstracts['sub:0'].print()
        # self.abstracts['keep_prob:0'].print()
        # self.abstracts['ConstantOfShape__684:0'].print()
        # self.abstracts['Concat__685:0'].print()
        print('=' * 10, "Summary", '=' * 10)
        print(f"Find {sum(item[1] for item in analyze_result_neg.items())} Negatives: {analyze_result_neg}")
        print(f"Find {sum(item[1] for item in analyze_result_pos.items())} Positives: {analyze_result_pos}")
        return self.possible_numerical_errors

    def forward(self, initial_abstracts, interp_config=dict()):
        """
            After the initial call of analyze(), this method tries to quickly relaunch the forward propagation based on initialized abstraction of start points
        :param initial_abstracts:
        :return:
        """
        interpreter = Interpreter(**interp_config)
        abstracts = initial_abstracts
        possible_numerical_errors = dict()

        optimize_op_filter = {"Mul",  # -> Square
                              "Sub",  # -> Zero
                              "Slice", "Concat", "Resize", "Sum"}

        for i, (node_name, ctx) in enumerate(self.queue):
            if ctx is None:
                if node_name not in abstracts:
                    print(f'! Initial node {node_name} lacks initialized abstracts')
            else:
                if node_name in abstracts:
                    # already obtained from previous op propagation, skip
                    pass
                else:
                    parent, now_node, ind_input, ind_output, node_optype, node_name, node = ctx



                    node_input_cnt = len(node.input)
                    unique_node_input_cnt = len(set([x for x in node.input]))
                    # detect potential optimizing operators
                    if unique_node_input_cnt != node_input_cnt and node.op_type not in optimize_op_filter:
                        print(f"({node.name}, {node.op_type}) can be optimized!\n{node.input}")
                    if node.op_type != 'Loop':
                        # optimize multiplication with two same inputs as square
                        if node.op_type == "Mul" and unique_node_input_cnt == 1:
                            cur_abst, cur_exceps = interpreter.handle(
                                [abstracts[node.input[0]]], None, "Square", node_name
                            )
                        else:
                            cur_abst, cur_exceps = interpreter.handle(
                                [abstracts[x] for x in node.input], node, node_optype, node_name
                            )
                    else:
                        # specially handle the loop dependencies
                        cur_abst, cur_exceps = interpreter.interp_Loop(
                            [abstracts[x] for x in node.input], node, node_optype, node_name,
                            loop_dependencies=dict(
                                [(x, abstracts[x]) for x in self.loop_dependencies[node.name] if
                                 x in abstracts])
                        )

                    # if node.op_type != 'Loop':
                    #     cur_abst, cur_exceps = interpreter.handle(
                    #         [abstracts[x] for x in node.input], node, node_optype, node_name
                    #     )
                    # else:
                    #     # specially handle the loop dependencies
                    #     cur_abst, cur_exceps = interpreter.interp_Loop(
                    #         [abstracts[x] for x in node.input], node, node_optype, node_name,
                    #         loop_dependencies=dict(
                    #             [(x, abstracts[x]) for x in self.loop_dependencies[node.name] if x in abstracts])
                    #     )

                    if len(cur_exceps) > 0:
                        roots = list()
                        for x in node.input:
                            if x in possible_numerical_errors:
                                roots.extend(possible_numerical_errors[x][1])
                        roots = set(roots)
                        # whether it is a catastrophic error
                        if len(roots) == 0:
                            roots = {node.name}
                            catastro = False
                        else:
                            catastro = True
                        possible_numerical_errors[now_node] = (cur_exceps, roots, catastro)
                    if cur_abst is None:
                        pass
                        # print(f'! No abstraction generated for {now_node}: '
                        #       f'node name = {node_name}, type = {node_optype}')
                    else:
                        if isinstance(cur_abst, Abstraction):
                            # single output node
                            abstracts[now_node] = cur_abst
                        else:
                            # multiple output node: execute once, update all output nodes
                            for i, cur_cur_abst in enumerate(cur_abst):
                                abstracts[node.output[i]] = cur_cur_abst

        return abstracts, possible_numerical_errors

    def gen_abstraction_heuristics(self, model_name):
        """
            Generate a dictionary which contains the heuristics for each initial tensor if necessary
        :return:
        """
        result = {}
        # load from DEBAR's specified ranges
        if model_name in SpecifiedRanges.specified_ranges:
            for name, values in SpecifiedRanges.specified_ranges[model_name].items():
                if len(values) == 2 and not isinstance(values[0], list):
                    values = [values]
                for i, value in enumerate(values):
                    if len(values) > 1:
                        new_name = name + "_%d:0" % i
                    else:
                        new_name = name + ":0"
                    result[new_name] = AbstractionInitConfig(diff=False, from_init=True,
                                                             lb=value[0] if value[0] is not None else
                                                             -PossibleNumericalError.OVERFLOW_LIMIT,
                                                             ub=value[1] if value[1] is not None else
                                                             PossibleNumericalError.OVERFLOW_LIMIT)

        # for name, values in self.initializer_dict.items():
        for name in self.start_points:
            if name == InterpModule.SUPER_START_NODE: continue
            dtype = self.signature_dict[name][0]
            if name in self.initializer_dict:
                data = self.initializer_dict[name][1]
            else:
                data = None

            if dtype not in discrete_types and name not in result:
                # print(name, np.min(data), np.max(data), data.shape)
                if (name.lower().count('dropout') > 0 and name.count('/') < 2) and (data is None or data.size == 1):
                    # looks like the div in dropout
                    print(
                        f'Parameter {name} looks like a dropout div, abstract by {AbstractionInitConfig.DROPOUT_CONFIG_DEFAULT}')
                    result[name] = AbstractionInitConfig(diff=False, from_init=False,
                                                         lb=AbstractionInitConfig.DROPOUT_CONFIG_DEFAULT[0],
                                                         ub=AbstractionInitConfig.DROPOUT_CONFIG_DEFAULT[1])
                elif (name.lower().count('keep_prob') > 0 and name.count('/') < 2) and (data is None or data.size == 1):
                    # looks like the keep_prob
                    print(
                        f'Parameter {name} looks like a keep_prob, abstract by {AbstractionInitConfig.KEEP_PROB_CONFIG_DEFAULT}')
                    result[name] = AbstractionInitConfig(diff=False, from_init=False,
                                                         lb=AbstractionInitConfig.KEEP_PROB_CONFIG_DEFAULT[0],
                                                         ub=AbstractionInitConfig.KEEP_PROB_CONFIG_DEFAULT[1])
                elif name.lower().count('moving_variance') > 0:
                    # looks like a variance
                    print(
                        f'Parameter {name} looks like a variance, abstract by {AbstractionInitConfig.VARIANCE_CONFIG_DEFAULT}')
                    result[name] = AbstractionInitConfig(diff=False, from_init=True,
                                                         lb=AbstractionInitConfig.VARIANCE_CONFIG_DEFAULT[0],
                                                         ub=AbstractionInitConfig.VARIANCE_CONFIG_DEFAULT[1])
                elif dtype in discrete_types:
                    print(f'Parameter or weight {name} corresponds to a discrete node, abstract by {AbstractionInitConfig.INT_CONFIG_DEFAULT}')
                    result[name] = AbstractionInitConfig(diff=False, from_init=False,
                                                         lb=AbstractionInitConfig.INT_CONFIG_DEFAULT[0],
                                                         ub=AbstractionInitConfig.INT_CONFIG_DEFAULT[1])
                elif data is not None and data.ndim >= 1 and data.size > 0 and np.max(data) - np.min(data) <= 1e-5 and abs(
                        np.max(data)) <= 1e-5:
                    # approaching zero tensor detected, overwrite
                    print(
                        f'Parameter {name} (shape: {data.shape}) is zero initialized, but may take over values --- abstract by {AbstractionInitConfig.WEIGHT_CONFIG_DEFAULT}')
                    result[name] = AbstractionInitConfig(diff=True, from_init=False,
                                                         lb=AbstractionInitConfig.WEIGHT_CONFIG_DEFAULT[0],
                                                         ub=AbstractionInitConfig.WEIGHT_CONFIG_DEFAULT[1])

        return result


def load_onnx_from_file(path, customize_shape=None):
    onnx_model = onnx.load_model(path)
    return InterpModule(onnx_model, customize_shape=customize_shape)
