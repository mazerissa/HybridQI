import torch
from torchvision import datasets, transforms
from src.models.mlp import MLP
from src.optimizers.adam import Adam 

tform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), 
                                                 (0.3081,)), 
                                                 transforms.Lambda(lambda x: x.view(-1))])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', 
                                                          train=True, 
                                                          download=True, 
                                                          transform=tform), 
                                                          batch_size=64, 
                                                          shuffle=True)

model = MLP()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for data, target in train_loader:
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} | Last Batch Loss: {loss.item():.4f}")