import torch

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad.data

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()