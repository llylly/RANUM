
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.graph_util import convert_variables_to_constants
from onnx import checker
import tf2onnx

def transform_assign_to_identity(graphdef):

    # print(dir(graphdef))
    empty_variables = dict()

    for node in graphdef.node:
        if node.op == 'Variable' or node.op == 'VariableV2' and len(node.input) == 0:
            print(node.attr)
            empty_variables[node.name] = node
    # print(empty_variables)

    to_replace = list()
    to_remove_names = set()
    to_remove = list()

    for node in graphdef.node:
        if node.op == 'Assign':
            if node.input[0] in empty_variables:
                to_replace.append(tf.NodeDef(name=node.input[0], op='identity', input=[node.input[1]], attr={'T': node.attr['T'], '_class': empty_variables[node.input[0]].attr['_class']}))
                if node.name not in to_remove_names:
                    to_remove_names.add(node.name)
                    to_remove.append(node)
                if empty_variables[node.input[0]].name not in to_remove_names:
                    to_remove_names.add(empty_variables[node.input[0]].name)
                    to_remove.append(empty_variables[node.input[0]])
                # print(node.name, 'to replace')
                # print(node)
            else:
                pass
                # print(node.name)

    # print(type(graphdef.node))
    # print(dir(graphdef.node))
    for item in to_remove:
        graphdef.node.remove(item)
    graphdef.node.extend(to_replace)

    return graphdef


def analyze_inputs_outputs(graphdef):
    inputs = list()
    # attrs_by_type = dict()

    outputs_set = set([node.name for node in graphdef.node])
    op_types = set()
    op_names = set([node.name for node in graphdef.node])

    print(op_names)

    for node in graphdef.node:
        if len(node.input) == 0 and (node.op != 'Const' or node.name.count('placeholder') + node.name.count('Placeholder') > 0):
            inputs.append(node.name)
        else:
            for input_tensor in node.input:
                if input_tensor in outputs_set:
                    outputs_set.remove(input_tensor)
        if node.op not in op_types:
            op_types.add(node.op)

        if node.name in ['Conv_2/biases/Adam', 'Conv_8/weights/Adam', 'Conv_3/biases/Adam_1/read', 'Conv/biases', 'fully_connected_2/weights', 'PlaceholderWithDefault/input'] or node.op in ['RandomUniform']:
            print(node.name)
            print(node.op)
            print(node.attr)
            print(node.attr['shape'])

    print(op_types)
    #
    #     # print(node.op)
    #     print(node.input)
    #     # print(node.attr)
    #     # print('')
    # for key in attrs_by_type:
    #     print('key:', key)
    #     print(attrs_by_type[key])
    outputs = list(outputs_set)
    return inputs, outputs


def parseProtoBuf(protobuf_path):

    print(tf.__version__)

    # import protobuf
    with open(protobuf_path) as f:
        txt = f.read()
        graph_def = text_format.Parse(txt, tf.GraphDef())

    print(type(graph_def))
    # print(graph_def.node)

    graph_def = transform_assign_to_identity(graph_def)

    input_nodes, outputs_nodes = analyze_inputs_outputs(graph_def)

    # # try:
    # #     with tf.Session() as sess:
    # #
    # #
    # #         tf.import_graph_def(graph_def, name="")
    # #         tf_graph = tf.get_default_graph()
    # #         input_nodes, outputs_nodes = analyze_inputs_outputs(tf_graph)
    # #
    # #         print('input nodes', input_nodes)
    # #         print('output nodes', outputs_nodes)
    # #         print([node.name for node in graph_def.node])
    # #
    # #         # print([v for v in tf.global_variables()])
    # #         #
    # #         # sess.run(tf.global_variables_initializer())
    # #         # frozen_tf_graph = convert_variables_to_constants(sess, graph_def, outputs_nodes)
    # # except Exception:
    # #     raise Exception('Error on freezing TF graph')
    # #
    # #
    # # # analyze input and output nodes
    # # # try:
    # # #     tf.import_graph_def(frozen_tf_graph, name="")
    # # #     tf_graph = tf.get_default_graph()
    # # #     input_nodes, outputs_nodes = analyze_inputs_outputs(tf_graph)
    # # #     print(input_nodes, outputs_nodes)
    # # # except Exception:
    # # #     raise Exception('Error on detecting input and output nodes')
    # #
    # # try:
    # #     onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph, input_names=input_nodes, output_names=outputs_nodes)
    # # except Exception:
    # #     raise Exception('Error on TF => ONNX stage')
    #
    #
    print('input nodes', input_nodes)
    print('output nodes', outputs_nodes)

    input_nodes = [x + ':0' for x in input_nodes]
    outputs_nodes = [x + ':0' for x in outputs_nodes]

    onnx_graph, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
                                                                      name=None, input_names=input_nodes, output_names=outputs_nodes, opset=None,
                                                                      custom_ops=None, custom_op_handlers=None, custom_rewriter=None,
                                                                      inputs_as_nchw=None, extra_opset=None,
                                                                      shape_override=None, target=None, large_model=False,
                                                                      output_path=None)
    #
    # print('-----')
    # print(onnx_graph)
    # print(type(onnx_graph))
    # onnx_model = onnx_graph.make_model("main")
    # print(onnx_model)
    # print(type(onnx_model))
    # checker.check_model(onnx_model)
    # print('Model is checked')
    # return onnx_model
    # pass

