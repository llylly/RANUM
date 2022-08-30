"""
    This script copies all .onnx files from model_zoo/grist_protobufs to model_zoo/girst_protobufs_onnx
"""

import sys
sys.path.append('.')
sys.path.append('..')
import shutil

from graphparser.grist_adaptor import current_new_scripts

if __name__ == '__main__':
    for id, s in enumerate(current_new_scripts):
        num = s.split('_')[1]
        shutil.copy(f'model_zoo/grist_protobufs/{s}/model.onnx', f'model_zoo/grist_protobufs_onnx/{num}.onnx')
    print('done')