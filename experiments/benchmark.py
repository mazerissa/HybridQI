import torch
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from src.models.mlp import MLP
from src.optimizers.sgd import SGD
from src.optimizers.momentum import Momentum
from src.optimizers.adam import Adam

# 1. Setup Data
tform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x: x.view(-1))])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=tform), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=tform), batch_size=1000)

def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for d, t in test_loader:
            correct += (model(d).argmax(1) == t).sum().item()
    return (correct / 10000) * 100

def run_experiment(opt_class, name, lr):
    print(f"\n🚀 Starting {name}...")
    model = MLP()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = opt_class(model.parameters(), lr=lr)
    
    losses = []
    start_time = time.time()
    
    for epoch in range(5):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            losses.append(loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
    
    end_time = time.time()
    acc = evaluate(model)
    
    return {
        "name": name,
        "losses": losses,
        "accuracy": acc,
        "time": end_time - start_time
    }

# 2. Run the Tournament
# Note: Adam usually needs a smaller LR than SGD
results = [
    run_experiment(SGD, "SGD", lr=0.01),
    run_experiment(Momentum, "Momentum", lr=0.01),
    run_experiment(Adam, "Adam", lr=0.001)
]

# 3. Print Final Leaderboard
print("\n" + "="*30)
print("🏆 FINAL BENCHMARK RESULTS")
print("="*30)
for r in results:
    print(f"{r['name']:<10} | Acc: {r['accuracy']:.2f}% | Time: {r['time']:.2f}s")

# 4. Plot Convergence Speed
plt.figure(figsize=(10, 6))
for r in results:
    plt.plot(r['losses'], label=r['name'], alpha=0.7)

plt.yscale('log') # Log scale helps see the difference when loss is small
plt.xlabel("Iterations (Batches)")
plt.ylabel("Loss (Log Scale)")
plt.title("Convergence Speed Comparison")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('visuals/optimizer_benchmark.png')
print("\n📊 Benchmark graph saved to visuals/optimizer_benchmark.png")