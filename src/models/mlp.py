import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        #Architecture
        self.funcition_01 = nn.Linear(784, 128)
        self.funcition_02 = nn.Linear(128, 64)
        self.funcition_03 = nn.Linear(62, 10)
        #Rectified Linear Unit or ReLU
        self.relu = nn.ReLU()

