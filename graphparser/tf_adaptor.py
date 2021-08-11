import os
import sys
sys.path.append('.')
sys.path.append('..')

from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.graph_util import convert_variables_to_constants
from onnx import helper, checker
import tf2onnx

from graphparser.utils import remove_node_and_decendents


def trim_nodes(graphdef, node_names):
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
            if pre_v in inv_edges:
                if node.name not in inv_edges[pre_v]:
                    inv_edges[pre_v].append(node.name)
            else:
                inv_edges[pre_v] = [node.name]

    que = node_names
    se = set(que)
    l = 0
    while l < len(que):
        if que[l] in inv_edges:
            for oute in inv_edges[que[l]]:
                if oute not in se:
                    se.add(oute)
                    que.append(oute)
        l += 1

    preversed_nodes = [node for node in graphdef.node if node.name not in se]
    del graphdef.node[:]
    graphdef.node.extend(preversed_nodes)

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
    name_mapping = dict()
    transform_node_list = list()
    possible_input_node_list = list()

    """STEP2: replace OneShotIterator or IteratorV2 by several variables and IteratorGetNext by several Identity"""
    # catch OneShotIterator
    find = True
    while find:
        find = False
        for node in graphdef.node:
            if node.op == 'OneShotIterator' or node.op == 'IteratorV2':

                for no in range(len(node.attr['output_shapes'].list.shape)):
                    new_shape = node.attr['output_shapes'].list.shape[no]
                    for i, item in enumerate(new_shape.dim):
                        if item.size == -1:
                            if i == 0:
                                item.size = 1
                            else:
                                # assume batch size = 8
                                item.size = 8
                    # if new_shape.dim[0].size == -1:
                    #     new_shape.dim[0].size = 1

                    new_node = tf.NodeDef(name=node.name + f'_{no}',
                                          op='VariableV2',
                                          input=[],
                                          attr={'container': node.attr['container'],
                                                'dtype': tf.AttrValue(type=node.attr['output_types'].list.type[no]),
                                                'shape': tf.AttrValue(shape=new_shape),
                                                'shared_name': node.attr['shared_name']})
                    to_extend.append(new_node)

                    if no == 0:
                        name_mapping[node.name] = new_node.name
                    else:
                        name_mapping[node.name + f':{no}'] = new_node.name
                    possible_input_node_list.append(new_node.name)

                find = True
                graphdef.node.remove(node)
                graphdef.node.extend(to_extend)
                to_extend = list()

    # catch IteratorGetNext
    find = True
    while find:
        find = False
        for node in graphdef.node:
            if node.op == 'IteratorGetNext':
                for no in range(len(node.attr['output_types'].list.type)):
                    new_node = tf.NodeDef(name=node.name + f'_{no}',
                                          op='Identity',
                                          input=[node.input[0] + f'_{no}'],
                                          attr={'T': tf.AttrValue(type=node.attr['output_types'].list.type[no])})
                    to_extend.append(new_node)

                    if no == 0:
                        name_mapping[node.name] = new_node.name
                    else:
                        name_mapping[node.name + f':{no}'] = new_node.name
                    possible_input_node_list.append(new_node.name)

                find = True
                graphdef.node.remove(node)
                graphdef.node.extend(to_extend)
                to_extend = list()

    # fix renamed operators
    for node in graphdef.node:
        node.input[:] = [item if item not in name_mapping else name_mapping[item] for item in node.input]
    name_mapping = dict()

    """STEP3: (semantic changed) shortcut the switch and propagate the input to both branches"""
    find = True
    while find:
        find = False
        for node in graphdef.node:
            if node.op == 'Switch':
                find = True
                # input_0: value; input_1: bool cond
                new_node0 = tf.NodeDef(name=node.name + '_0',
                                       op='Identity',
                                       input=[node.input[0]],
                                       attr={'T': node.attr['T']})
                new_node1 = tf.NodeDef(name=node.name + '_1',
                                       op='Identity',
                                       input=[node.input[0]],
                                       attr={'T': node.attr['T']})
                name_mapping[node.name] = node.name + '_0'
                name_mapping[node.name + ':1'] = node.name + '_1'
                graphdef.node.remove(node)
                graphdef.node.extend([new_node0, new_node1])

    """STEP4: (semantic changed) shortcut RandomShuffle as Identity"""
    for node in graphdef.node:
        if node.op == 'RandomShuffle':
            node.op = 'Identity'
            del node.attr['seed']
            del node.attr['seed2']

    """STEP4b: change ParseExampleSingle to VariableV2 (caution: may separate some nodes in graph)"""
    """ tried to fix ptn.pbtxt, but failed: Field 'shape' of type is required but missing. """
    # find = True
    # while find:
    #     find = False
    #     for node in graphdef.node:
    #         if node.op == 'ParseSingleExample':
    #             find = True
    #             new_nodes = list()
    #             for i in range(len(node.attr['dense_shapes'].list.shape)):
    #                 new_node = tf.NodeDef(name=node.name + f'_{i}',
    #                                       op='VariableV2',
    #                                       input=[],
    #                                       attr={
    #                                           'container': tf.AttrValue(s=b''),
    #                                           'dtype': tf.AttrValue(type=node.attr['Tdense'].list.type[i]),
    #                                           'shape': tf.AttrValue(shape=node.attr['dense_shapes'].list.shape[i]),
    #                                           'shared_name': tf.AttrValue(s=b'')
    #                                       })
    #                 if i == 0:
    #                     name_mapping[node.name] = new_node.name
    #                 else:
    #                     name_mapping[node.name + f':{i}'] = new_node.name
    #                 new_nodes.append(new_node)
    #             graphdef.node.remove(node)
    #             graphdef.node.extend(new_nodes)


    """STEP5: split ShapeN node to several Shape node since shabby onnx converter does not support ShapeN"""
    find = True
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

    # fix renamed operators
    for node in graphdef.node:
        node.input[:] = [item if item not in name_mapping else name_mapping[item] for item in node.input]

    """STEP6: transform Variable or VariableV2 to constant"""
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
                    raise Exception(f"unusual variable node {item.name}: shared_name field is not empty")
            elif field == '_output_shapes':
                pass
            else:
                raise Exception(f"unusual variable node {item.name}: new field: {field}")

        try:
            arr_shape = [item.size for item in value_field['shape'].shape.dim]
            if value_field['dtype'].type in [dtypes.int16, dtypes.int32, dtypes.int64, dtypes.int8, dtypes.uint16, dtypes.uint32, dtypes.uint64, dtypes.uint8]:
                initial_value = 0
            elif value_field['dtype'].type in [dtypes.string]:
                initial_value = ''
            elif value_field['dtype'].type in [dtypes.bool]:
                initial_value = False
            else:
                initial_value = 0.
            tfvalue = tf.AttrValue(tensor=tf.make_tensor_proto(values=initial_value, dtype=value_field['dtype'].type, shape=arr_shape))
            new_node_attrs['value'] = tfvalue
        except:
            raise Exception('unable to make tfvalue: maybe dtype or shape fields are missing or illegal')

        new_node = tf.NodeDef(name=item.name, op='Const', input=[], attr=new_node_attrs)

        to_extend.append(new_node)

    for item in to_remove:
        graphdef.node.remove(item)
    graphdef.node.extend(to_extend)

    """STEP7: Rename some operations to freeze the graph"""
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

    """STEP8: remove the annoying ApplyAdam, BroadcastGradientArgs, ConcatOffset, IteratorToStringHandle, SaveV2, RestoreV2, etc nodes and their descendents"""
    # these types of nodes are gradient related, i.e., training related, or just for printing or saving weights
    # so normally trimming them will not affect our analysis precision for inference time applications
    # moreover, they are not supported by onnx or static graph

    # NOTE: principally we should not ignore Queue related ops since they read in the input data, but somehow tf2onnx does not support it so we give up...

    node_to_trim = list()
    for node in graphdef.node:
        if node.op == 'ApplyAdam' \
                or node.op == 'BroadcastGradientArgs' or node.op == 'ConcatOffset' \
                or node.op == 'ScalarSummary' or node.op == 'ImageSummary' or node.op == 'HistogramSummary' or node.op == 'Print' \
                or node.op == 'IteratorToStringHandle' \
                or node.op == 'SaveV2' or node.op == 'ShardedFilename' or node.op == 'StringJoin' \
                or node.op == 'RestoreV2' or node.op == 'RestoreSlice'\
                or node.op == 'IsVariableInitialized' \
                or node.op == 'GeneratorDataset' or node.op == 'TensorSliceDataset' \
                or node.op == 'PaddingFIFOQueueV2' or node.op == 'RandomShuffleQueueV2' \
                or node.op == 'QueueSizeV2' or node.op == 'TFRecordReaderV2' or node.op == 'ReaderReadV2' \
                or node.op == 'ApplyRMSProp' \
                or node.op == 'Assert' \
                or node.op == 'Lgamma' \
                or node.op == 'ScatterUpdate' \
                or node.op == 'L2Loss' or node.op == 'PyFunc' \
                or node.op == 'Rank' \
                or node.op == 'WholeFileReaderV2' or node.op == 'TextLineReaderV2' \
                or node.name.count('gradient') > 0:
            node_to_trim.append(node.name)
    graphdef = trim_nodes(graphdef, node_to_trim)

    """STEP9: replace PlaceholderWithDefault by its Const input"""
    find = True
    while find:
        find = False
        for node in graphdef.node:
            if node.op == 'PlaceholderWithDefault' and [snode for snode in graphdef.node if snode.name == node.input[0]][0].op == 'Const':
                find = True
                # assert that it is a normal placeholderwithdefault
                assert len(node.input) == 1
                default_node_name = node.input[0]

                default_node = [node for node in graphdef.node if node.name == default_node_name][0]
                # assert that the default value node is normal Const node
                assert default_node.op == 'Const' and len(default_node.input) == 0

                default_node.name = node.name
                graphdef.node.remove(node)

    """STEP10: replace TruncatedNormal by StatelessRandomNormal since onnx does not support truncated normal distribution"""
    """Reminder: for boundedness, for all normally distributed nodes, we view its range is [mean - 2 * stddev, mean + 2 * stddev], i.e., view them all as truncated_normal"""
    # new_nodes = list()
    for node in graphdef.node:
        if node.op == 'TruncatedNormal':
            node.op = 'RandomStandardNormal'

    """STEP11: replace log1p = ln (1 + x) by log with add"""
    for node in graphdef.node:
        if node.op == 'Log1p':
            node_name = node.name
            const_one_node = tf.NodeDef(name=node_name + '/ConstOne',
                                        op='Const',
                                        attr={
                                            'dtype': node.attr['T'],
                                            'value': tf.AttrValue(tensor=tf.make_tensor_proto(values=1., dtype=dtypes.float32, shape=[1,]))
                                        })
            add_node = tf.NodeDef(name=node_name + '/Add',
                                  op='Add',
                                  input=[node.input[0], node_name + '/ConstOne'],
                                  attr={'T': node.attr['T']})
            node.op = 'Log'
            node.input[0] = node_name + '/Add'
            graphdef.node.extend([const_one_node, add_node])

    """STEP12: trim the transform_node_list because some of them might be removed in later steps"""
    node_names = set([node.name for node in graphdef.node])
    transform_node_list = [item for item in transform_node_list if item in node_names]
    possible_input_node_list = [item for item in possible_input_node_list if item in node_names]

    """STEP13: remove the colocated requirement if the required node no longer exists"""
    for node in graphdef.node:
        if '_class' in node.attr:
            source = node.attr['_class'].list.s[0].decode('utf-8')
            if source.startswith('loc:@'):
                source = source[5:]
                if source not in node_names:
                    del node.attr['_class']

    return graphdef, transform_node_list, possible_input_node_list


def analyze_inputs_outputs(graphdef):
    inputs = list()

    outputs_set = set([node.name for node in graphdef.node if node.op != 'NoOp' and node.op != 'Save' and
                       node.op != 'SaveV2' and node.op != 'SaveSlices'
                       and node.op != 'Assert' and node.op != 'QueueEnqueueManyV2' and node.op != 'QueueEnqueueV2' and node.op != 'QueueCloseV2'
                       and node.op != 'LookupTableImportV2'])
    op_types = set()
    op_names = set([node.name for node in graphdef.node])

    for node in graphdef.node:
        if len(node.input) == 0 and (node.op != 'Const' or node.name.count('placeholder') + node.name.count('Placeholder')
                                     + node.name.count('input') + node.name.count('Input') > 0) \
                and node.op != 'NoOp' and node.op != 'Save' and node.op != 'SaveV2' and node.op != 'SaveSlices'\
                and node.op != 'Assert' and node.op != 'QueueEnqueueManyV2' and node.op != 'QueueCloseV2':
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

    print('TF version =', tf.__version__)

    # import protobuf
    with open(protobuf_path) as f:
        txt = f.read()
        graph_def = text_format.Parse(txt, tf.GraphDef())

    # print(type(graph_def))

    graph_def, variable_node_list, possible_input_node_list = freeze_and_initialize_graph(graph_def)

    # print('finished graph freezing')

    input_nodes, outputs_nodes = analyze_inputs_outputs(graph_def)
    input_nodes = list(set(input_nodes).union(set(possible_input_node_list)))

    # print('input nodes', input_nodes)
    # print('output nodes', outputs_nodes)

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

    # post-processing
    # need to remove all DynamicStitch and Merge nodes:
    # if there is such node, means the translation did not go well and we have to remove the node and its decendents
    find = True
    while find:
        find = False
        for node in onnx_graph.graph.node:
            if node.op_type == 'DynamicStitch' or node.op_type == 'Merge':
                onnx_graph = remove_node_and_decendents(onnx_graph, node.name)
                find = True
                break


    try:
        checker.check_model(onnx_graph)
    except checker.ValidationError as e:
        raise Exception('The model is invalid: %s' % e)
    else:
        pass
        # print('The model is valid!')

    return onnx_graph, {'variable nodes': variable_node_list, 'narrow inputs': possible_input_node_list, 'broad inputs': input_nodes}

# ====== the following dumps the support pbtxt to onnx format ======


import os
import json

banned_list = ['compression_entropy_coder', 'deep_speech', 'delf', 'domain_adaptation', 'faster_rcnn_resnet_50',
               'feelvos', 'fivo_ghmm', 'fivo_srnn', 'fivo_vrnn', 'gan_cifar', 'gan_mnist',
               'learning_to_remember_rare_events', 'neural_gpu1', 'neural_gpu2',
               'textsum', 'ptn', 'sentiment_analysis', 'skip_thought',
               'video_prediction']

# debug
# permit_list = ['adversarial_crypto']


def convert_protobuf_file(file_path):
    print(f'converting file {file_path}')
    model, annotation = parseProtoBuf(file_path)
    print('model type is', type(model))
    print('# variable nodes:', len(annotation['variable nodes']))
    print('# narrow inputs:', len(annotation['narrow inputs']))
    print('# broad inputs:', len(annotation['broad inputs']))

    bare_name = os.path.basename(file_path)
    with open(f'model_zoo/tf_protobufs_onnx/{bare_name}.onnx', 'wb') as f:
        f.write(model.SerializeToString())
    with open(f'model_zoo/tf_protobufs_onnx/{bare_name}.json', 'w') as f:
        json.dump(annotation, f, indent=2)


def convert_protobuf_folder(dir_path='model_zoo/tf_protobufs'):
    files = [item for item in os.listdir(dir_path) if item.endswith('.pbtxt')]
    files = sorted(files)
    succeed = list()
    failed = list()
    skipped = list()
    for file in files:
        try:
            if len([item for item in banned_list if file.count(item) > 0]) == 0:
                convert_protobuf_file(os.path.join(dir_path, file))
                succeed.append(file)
            else:
                skipped.append(file)
        except Exception as e:
            print(f'Error (type is {type(e)}):', str(e))
            failed.append(file)

    assert len(failed) == 0
    print(f'Success on {len(succeed)}; failed (anticipated) on {len(skipped)}')

if __name__ == '__main__':
    convert_protobuf_folder()
