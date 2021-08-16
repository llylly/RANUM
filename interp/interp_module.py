
import torch
from torch.nn import Module
import onnx


class InterpModule():

    def __init__(self, onnx_model):
        self.onnx_model = onnx_model



def load_onnx_from_file(path):
    onnx_model = onnx.load_model(path)

    return InterpModule(onnx_model)


def analyze(module: InterpModule):
    pass