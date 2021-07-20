import os

from graphparser import protobuf_adaptor


def test_protobuf_file(file_path):
    print(f'testing file {file_path}')
    model = protobuf_adaptor.parseProtoBuf(file_path)
    print('model type is', type(model))


def test_protobuf_folder(dir_path='model_zoo/tf_protobufs'):
    files = [item for item in os.listdir(dir_path) if item.endswith('.pbtxt')]
    files = sorted(files)
    for file in files:
        test_protobuf_file(os.path.join(dir_path, file))

if __name__ == '__main__':
    test_protobuf_folder()