import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for classification tasks.
    
    Architecture:
    - Input: 784 features (28x28 flattened images)
    - Hidden Layer 1: 128 neurons with ReLU
    - Hidden Layer 2: 64 neurons with ReLU
    - Output: 10 classes
    """
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 784)
            
        Returns:
            Output logits of shape (batch_size, 10)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

