import os
import sys
sys.path.append('.')

from graphparser import tf_adaptor

# it's hard to let ONNX support the queue structure of compression_entropy_coder
# deep_speech's shape somehow requires initialized value to infer from
# delf contains DecodeJpeg node
# domain_adaptation contains illegal tensor_content
# faster_rcnn_resnet_50 contains many Switch node
# feelvos is large, and says ValueError: Node 'map/while/case/cond/DecodeRaw' has an _output_shapes attribute inconsistent with the GraphDef for output #0: Shapes must be equal rank, but are 1 and 3
# fivo_ghmm, fivo_srnn, fivo_vrnn: ValueError: Dimension 0 in both shapes must be equal, but are 16 and 4. Shapes are [16] and [4]. for 'while_1/Select' (op: 'Select') with input shapes: [?], [16], [4].
# gan_cifar, gan_mnist: existance of string_ref
# gan_mnist: The model is invalid: Field 'shape' of type is required but missing.
# learning_to_remember_rare_events: Unsupported ops: Counter({'Switch': 7})
# neural_gpu1, neural_gpu2: ValueError: squeeze_dims[0] not in [0,0). for 'gpu0/Squeeze' (op: 'Squeeze') with input shapes: [].
# textsum model is too large
# ptn: Field 'shape' of type is required but missing.
# sbn: has dynamic rank
# sentiment_analysis: contain many VarHandleOp, which is not easy to be transformed to Const
# skip_thought: ValueError: Graph has cycles, node=decoder_post/decoder_post/while/Merge
# video_prediction: cannot infer many shapes. Exception: The model is invalid: Field 'shape' of type is required but missing.
banned_list = ['compression_entropy_coder', 'deep_speech', 'delf', 'domain_adaptation', 'faster_rcnn_resnet_50',
               'feelvos', 'fivo_ghmm', 'fivo_srnn', 'fivo_vrnn', 'gan_cifar', 'gan_mnist',
               'learning_to_remember_rare_events', 'neural_gpu1', 'neural_gpu2',
               'textsum', 'ptn', 'sentiment_analysis', 'skip_thought',
               'video_prediction']
permit_list = []

# debug
# permit_list = ['adversarial_crypto']


def test_protobuf_file(file_path):
    print(f'testing file {file_path}')
    model, annotation = tf_adaptor.parseProtoBuf(file_path)
    print('model type is', type(model))
    print('# variable nodes:', len(annotation['variable nodes']))
    print('# narrow inputs:', len(annotation['narrow inputs']))
    print('# broad inputs:', len(annotation['broad inputs']))


def test_protobuf_folder(dir_path='model_zoo/tf_protobufs'):
    files = [item for item in os.listdir(dir_path) if item.endswith('.pbtxt')]
    files = sorted(files)
    succeed = list()
    failed = list()
    skipped = list()
    for file in files:
        try:
            if len([item for item in banned_list if file.count(item) > 0]) == 0 \
                    and (len(permit_list) == 0 or len([item for item in permit_list if file.count(item) > 0]) > 0):
                test_protobuf_file(os.path.join(dir_path, file))
                succeed.append(file)
            else:
                skipped.append(file)
        except Exception as e:
            print(str(e))
            failed.append(file)

    assert len(failed) == 0
    print(f'Success on {len(succeed)}; failed (anticipated) on {len(skipped)}')

if __name__ == '__main__':
    test_protobuf_folder()
