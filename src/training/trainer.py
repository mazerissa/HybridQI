import torch
from torchvision import datasets, transforms
from src.models.mlp import MLP

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: torch.flatten(x))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
model = MLP()
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.01

for epoch in range(5):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
            model.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")