import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        #Architecture
        self.funcition_01 = nn.Linear(784, 128)
        self.funcition_02 = nn.Linear(128, 64)
        self.funcition_03 = nn.Linear(64, 10)
        #Rectified Linear Unit or ReLU
        self.relu = nn.ReLU()

    def forward(self, m):
        m = self.relu(self.funcition_01(m))
        m = self.relu(self.funcition_02(m))
        m = self.funcition_03(m)
        return m
    

