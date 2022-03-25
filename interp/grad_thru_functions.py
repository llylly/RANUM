"""
    This module contains several overwritten PyTorch functions that provide more stable gradient signs.
    The forward pass is kept the same as original functions or tight interval abstractions,
    while the gradient part is mostly overwritten towards the purpose of preserving gradient signs.
"""
import torch

from interp.interp_utils import POSTIVE_MINIMUM

class StraightSigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.sigmoid()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straight-through
        return grad_output


class StraightTanh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.tanh()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straightthrough
        return grad_output


class StraightSoftmaxInterval(torch.autograd.Function):

    @staticmethod
    def forward(ctx, lb, ub, multiplies, axis):
        exp_lb = torch.exp(lb - torch.max(ub, dim=axis, keepdim=True)[0])
        exp_ub = torch.exp(ub - torch.max(ub, dim=axis, keepdim=True)[0])
        # inputs: [l1, l2, l3], [u1, u2, u3]
        # softmax_lb = [l1 / (l1 + u2 + u3), ...]
        # softmax_ub = [u1 / (u1 + l2 + l3)]
        # lb = exp_lb / (torch.sum(exp_ub * multiplies, dim=axis, keepdim=True) - exp_ub + exp_lb)
        # ub = exp_ub / (torch.sum(exp_lb * multiplies, dim=axis, keepdim=True) - exp_lb + exp_ub)
        lb = exp_lb / torch.maximum((torch.sum(exp_ub * multiplies, dim=axis, keepdim=True) - exp_ub + exp_lb), torch.full_like(exp_lb, POSTIVE_MINIMUM, device=exp_lb.device))
        ub = exp_ub / (torch.sum(exp_lb * multiplies, dim=axis, keepdim=True) - exp_lb + exp_ub)
        ub = torch.where(torch.isnan(ub), torch.ones_like(ub, device=ub.device), ub)
        return lb, ub

    @staticmethod
    def backward(ctx, grad_lb, grad_ub):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straightthrough
        # print('work')
        # return grad_lb - grad_ub, -grad_lb + grad_ub, None, None
        # return grad_lb + grad_ub, grad_lb + grad_ub, None, None
        return grad_lb, grad_ub, None, None

class StraightSoftmaxIntervalLb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, lb, ub, multiplies, axis):
        exp_lb = torch.exp(lb - torch.max(ub, dim=axis, keepdim=True)[0])
        exp_ub = torch.exp(ub - torch.max(ub, dim=axis, keepdim=True)[0])
        # inputs: [l1, l2, l3], [u1, u2, u3]
        # softmax_lb = [l1 / (l1 + u2 + u3), ...]
        # softmax_ub = [u1 / (u1 + l2 + l3)]
        lb = exp_lb / (torch.sum(exp_ub * multiplies, dim=axis, keepdim=True) - exp_ub + exp_lb)
        return lb

    @staticmethod
    def backward(ctx, grad_lb):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straightthrough
        # print('work')
        return grad_lb, -grad_lb, None, None


class StraightSoftmaxIntervalUb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, lb, ub, multiplies, axis):
        exp_lb = torch.exp(lb - torch.max(ub, dim=axis, keepdim=True)[0])
        exp_ub = torch.exp(ub - torch.max(ub, dim=axis, keepdim=True)[0])
        # inputs: [l1, l2, l3], [u1, u2, u3]
        # softmax_lb = [l1 / (l1 + u2 + u3), ...]
        # softmax_ub = [u1 / (u1 + l2 + l3)]
        ub = exp_ub / (torch.sum(exp_lb * multiplies, dim=axis, keepdim=True) - exp_lb + exp_ub)
        return ub

    @staticmethod
    def backward(ctx, grad_ub):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straightthrough
        # print('work')
        return -grad_ub, grad_ub, None, None


class StraightRelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.relu()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straightthrough
        input, = ctx.saved_tensors
        return torch.where(input >= 0., grad_output, 0.01 * grad_output)


class SmoothRelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.relu()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straightthrough
        input, = ctx.saved_tensors
        return torch.where(input >= 5., (1.-torch.exp(-torch.clip(input, min=5.))) * grad_output,
                                         (1. - 1. / (1. + torch.exp(torch.clip(input, max=5.+1e-5)))) * grad_output)


class StraightExp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = input.exp()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straight-through
        output, = ctx.saved_tensors
        # gradient clip to avoid vanishing gradients
        cliped_output = torch.clip(output, 0.01, 1e+5)
        return grad_output * cliped_output


class StraightMinimum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_a, input_b):
        return torch.minimum(input_a, input_b)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straightthrough
        return grad_output, grad_output


class StraightMaximum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_a, input_b):
        return torch.maximum(input_a, input_b)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # straightthrough
        return grad_output, grad_output