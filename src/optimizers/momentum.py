import torch

class Momentum:
    def __init__(self, parameters, lr=0.01, beta=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta = beta
        self.velocities = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                self.velocities[i] = self.beta * self.velocities[i] + p.grad.data
                p.data -= self.lr * self.velocities[i]

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()