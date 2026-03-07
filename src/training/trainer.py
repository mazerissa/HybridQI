import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional


class Trainer:
    """
    A modular trainer class for training PyTorch models.
    
    Features:
    - Support for multiple optimizers
    - Custom loss functions
    - Multiple devices (CPU/CUDA)
    - Comprehensive metric tracking
    - Training and evaluation modes
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        loss_fn: nn.Module,
        device: str = "cpu",
        epochs: int = 5
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance (SGD, Adam, Momentum, etc.)
            loss_fn: Loss function
            device: Device to run on ('cpu' or 'cuda')
            epochs: Number of training epochs
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.history = {
            "train_losses": [],
            "train_accuracies": [],
            "test_accuracies": [],
            "batch_losses": []
        }

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            self.history["batch_losses"].append(loss_val)
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        self.history["train_losses"].append(avg_loss)
        return avg_loss

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Accuracy percentage (0-100)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        accuracy = (correct / total) * 100
        return accuracy
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: Optional DataLoader for test data
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(train_loader)

            if verbose:
                msg = f"Epoch {epoch + 1}/{self.epochs} | Avg Loss: {avg_loss:.4f}"
                
                if test_loader is not None:
                    test_acc = self.evaluate(test_loader)
                    self.history["test_accuracies"].append(test_acc)
                    msg += f" | Test Acc: {test_acc:.2f}%"
                
                print(msg)
        
        return self.history
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with loss and accuracy history
        """
        return self.history