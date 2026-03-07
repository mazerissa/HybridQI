import numpy as np

class Context:
    def __init__(self, op, *parents):
        self.op = op
        self.parents = parents
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

class Add:
    @staticmethod
    def forward(ctx, a, b):
        return a + b
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class Mul:
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a
    
class ReLU:
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.maximum(0, a)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_input = grad_output * (a > 0)
        return grad_input