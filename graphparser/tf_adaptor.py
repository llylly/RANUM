
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.graph_util import convert_variables_to_constants
from onnx import helper, checker
import tf2onnx

from graphparser.utils import remove_node_and_decendents


def trim_node(graphdef, node_name):
    """
        remove one node and all nodes that rely on it via BFS
    :param graphdef:
    :param node_name:
    :return:
    """
    inv_edges = dict()
    for node in graphdef.node:
        for pre_v in node.input:
            if pre_v.startswith('^'):
                pre_v = pre_v[1:]
            if pre_v.count(':') > 0:
                pre_v = pre_v[:pre_v.index(':')]
            if pre_v in inv_edges and node.name not in inv_edges[pre_v]:
                inv_edges[pre_v].append(node.name)
            else:
                inv_edges[pre_v] = [node.name]

    que = [node_name]
    se = set(que)
    l = 0
    while l < len(que):
        if que[l] in inv_edges:
            for oute in inv_edges[que[l]]:
                if oute not in se:
                    se.add(oute)
                    que.append(oute)
        l += 1

    to_remove = [node for node in graphdef.node if node.name in se]
    for item in to_remove:
        graphdef.node.remove(item)

    return graphdef


def freeze_and_initialize_graph(graphdef):
    """
        Freeze the graph that only preserves the architecture information - the default graph freezing in TF requires
    :param graphdef:
    :return: (transformed graph def, transformed node list)
    """

    """STEP 1: convert all Variable to Constant, and record their names in the mean time"""
    to_remove = list()
    to_extend = list()
    transform_node_list = list()
    possible_input_node_list = list()

    for node in graphdef.node:
        if node.op == 'Variable' or node.op == 'VariableV2':
            to_remove.append(node)
            transform_node_list.append(node.name)
        if node.op == 'Placeholder' or node.op == 'PlaceholderWithDefault':
            possible_input_node_list.append(node.name)

    for item in to_remove:

        # a variable node should not have input nodes
        try:
            assert len(item.input) == 0
        except:
            raise Exception("unusual variable node: the input is not empty")

        new_node_attrs = dict()
        value_field = dict()

        for field in item.attr:
            if field == '_class':
                new_node_attrs[field] = item.attr[field]
            elif field == 'container':
                try:
                    assert item.attr[field].s == b''
                except:
                    raise Exception("unusual variable node: container field is not empty")
            elif field == 'dtype':
                value_field[field] = item.attr[field]
                new_node_attrs[field] = item.attr[field]
            elif field == 'shape':
                value_field[field] = item.attr[field]
            elif field == 'shared_name':
                try:
                    assert item.attr[field].s == b''
                except:
                    raise Exception("unusual variable node: shared_name field is not empty")
            else:
                raise Exception(f"unusual variabel node: new field: {field}")

        try:
            arr_shape = [item.size for item in value_field['shape'].shape.dim]
            tfvalue = tf.AttrValue(tensor=tf.make_tensor_proto(values=0., dtype=value_field['dtype'].type, shape=arr_shape))
            new_node_attrs['value'] = tfvalue
        except:
            raise Exception('unable to make tfvalue: maybe dtype or shape fields are missing or illegal')

        new_node = tf.NodeDef(name=item.name, op='Const', input=[], attr=new_node_attrs)

        to_extend.append(new_node)

    for item in to_remove:
        graphdef.node.remove(item)
    graphdef.node.extend(to_extend)

    """STEP2: Rename some operations to feeze the graph"""
    # copy from https://github.com/onnx/tensorflow-onnx/issues/77#issuecomment-445066091
    for node in graphdef.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]

    """STEP3: remove the annoying ApplyAdam, BroadcastGradientArgs, ConcatOffset nodes and their descendents"""
    # these three types of nodes are gradient related, i.e., training related
    # so normally trimming them will not affect our analysis precision for inference time applications
    # moreover, they are not supported by onnx or static graph
    find = True
    while find:
        find = False
        for node in graphdef.node:
            if node.op == 'ApplyAdam' or node.op == 'BroadcastGradientArgs' or node.op == 'ConcatOffset'\
                    or node.op == 'ScalarSummary':
                find = True
                graphdef = trim_node(graphdef, node.name)
                break

    """STEP4: replace PlaceholderWithDefault by its Const input"""
    find = True
    while find:
        find = False
        for node in graphdef.node:
            if node.op == 'PlaceholderWithDefault':
                find = True
                # assert that it is a normal placeholderwithdefault
                assert len(node.input) == 1
                default_node_name = node.input[0]

                default_node = [node for node in graphdef.node if node.name == default_node_name][0]
                # assert that the default value node is normal Const node
                assert default_node.op == 'Const' and len(default_node.input) == 0

                default_node.name = node.name
                graphdef.node.remove(node)

    """STEP5: replace OneShotIterator by a zero const, and record its name to possible input list"""
    # TODO

    """STEP6: split ShapeN node to several Shape node since shabby onnx converter does not support ShapeN"""
    find = True
    name_mapping = dict()
    while find:
        find = False
        for node in graphdef.node:
            if node.op == 'ShapeN':
                find = True
                new_nodes = list()
                for i, now_input in enumerate(node.input):
                    node_name = node.name if i == 0 else node.name + f'_{i}'
                    if i > 0:
                        name_mapping[node.name + f':{i}'] = node.name + f'_{i}'
                    new_node = tf.NodeDef(name=node_name,
                                          op='Shape',
                                          input=[now_input],
                                          attr={'T': node.attr['T'], 'out_type': node.attr['out_type']})
                    new_nodes.append(new_node)
                graphdef.node.remove(node)
                graphdef.node.extend(new_nodes)

    """STEP7: replace TruncatedNormal by StatelessRandomNormal since onnx does not support truncated normal distribution"""
    """Reminder: for boundedness, for all normally distributed nodes, we view its range is [mean - 2 * stddev, mean + 2 * stddev], i.e., view them all as truncated_normal"""
    # new_nodes = list()
    for node in graphdef.node:
        if node.op == 'TruncatedNormal':
            node.op = 'RandomStandardNormal'

    """STEP8: fix renamed operators"""
    for node in graphdef.node:
        node.input[:] = [item if item not in name_mapping else name_mapping[item] for item in node.input]

    """STEP9: trim the transform_node_list because some of them might be removed in later steps"""
    node_names = set([node.name for node in graphdef.node])
    transform_node_list = [item for item in transform_node_list if item in node_names]

    return graphdef, transform_node_list, possible_input_node_list


def analyze_inputs_outputs(graphdef):
    inputs = list()

    outputs_set = set([node.name for node in graphdef.node if node.op != 'NoOp'])
    op_types = set()
    op_names = set([node.name for node in graphdef.node])

    for node in graphdef.node:
        if len(node.input) == 0 and (node.op != 'Const' or node.name.count('placeholder') + node.name.count('Placeholder') > 0):
            inputs.append(node.name)
        else:
            for input_tensor in node.input:
                if input_tensor in outputs_set:
                    outputs_set.remove(input_tensor)
        if node.op not in op_types:
            op_types.add(node.op)

    outputs = list(outputs_set)
    return inputs, outputs


def parseProtoBuf(protobuf_path):

    print(tf.__version__)

    # import protobuf
    with open(protobuf_path) as f:
        txt = f.read()
        graph_def = text_format.Parse(txt, tf.GraphDef())

    print(type(graph_def))

    graph_def, variable_node_list, possible_input_node_list = freeze_and_initialize_graph(graph_def)

    input_nodes, outputs_nodes = analyze_inputs_outputs(graph_def)

    print('input nodes', input_nodes)
    print('output nodes', outputs_nodes)

    input_nodes = [x + ':0' if x.count(':') == 0 else x for x in input_nodes]
    outputs_nodes = [x + ':0' if x.count(':') == 0 else x for x in outputs_nodes]

    onnx_graph, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
                                         name=None, input_names=input_nodes, output_names=outputs_nodes, opset=13,
                                         custom_ops=None, custom_op_handlers=None, custom_rewriter=None,
                                         inputs_as_nchw=None, extra_opset=None,
                                         shape_override=None, target=None, large_model=False,
                                         output_path=None)

    # it should not use external tensor storage
    assert external_tensor_storage is None

    # post-process
    # need to remove all DynamicStitch nodes:
    # if there is such node, means the translation did not go well and we have to remove the node and its decendents
    find = True
    while find:
        find = False
        for node in onnx_graph.graph.node:
            if node.op_type == 'DynamicStitch':
                onnx_graph = remove_node_and_decendents(onnx_graph, node.name)
                find = True
                break

    try:
        checker.check_model(onnx_graph)
    except checker.ValidationError as e:
        raise Exception('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    return onnx_graph

