import torch
from torchvision import datasets, transforms
from src.models.mlp import MLP

tform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,)), 
    transforms.Lambda(lambda x: x.view(-1))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=tform), 
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=tform), 
    batch_size=1000
)

model = MLP()
loss_fn = torch.nn.CrossEntropyLoss()
lr = 0.01

for epoch in range(5):
    for data, target in train_loader:
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
            model.zero_grad()
            
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    correct = sum((model(d).argmax(1) == t).sum().item() for d, t in test_loader)
    print(f"Test Accuracy: {correct / 100:.2f}%")