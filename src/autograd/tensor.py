import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True, _ctx=None):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._ctx = _ctx

    def backward(self):
        from src.autograd.engine import backward_engine
        backward_engine(self)

    def __add__(self, other):
        from src.autograd.operations import Add, Context
        other = other if isinstance(other, Tensor) else Tensor(other)
        ctx = Context(Add, self, other)
        return Tensor(Add.forward(ctx, self.data, other.data), _ctx=ctx)

    def __mul__(self, other):
        from src.autograd.operations import Mul, Context
        other = other if isinstance(other, Tensor) else Tensor(other)
        ctx = Context(Mul, self, other)
        return Tensor(Mul.forward(ctx, self.data, other.data), _ctx=ctx)

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"
    
    def relu(self):
        from src.autograd.operations import ReLU, Context
        ctx = Context(ReLU, self)
        return Tensor(ReLU.forward(ctx, self.data), _ctx=ctx)