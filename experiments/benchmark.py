import torch
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pathlib import Path
from src.models.mlp import MLP
from src.optimizers.sgd import SGD
from src.optimizers.momentum import Momentum
from src.optimizers.adam import Adam

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path("visuals").mkdir(exist_ok=True)

tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=tform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=tform),
    batch_size=TEST_BATCH_SIZE
)

def evaluate(model):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            preds = model(data).argmax(dim=1)
            correct += (preds == target).sum().item()

    return (correct / len(test_loader.dataset)) * 100


def run_experiment(opt_class, name, lr):
    print(f"\nStarting {name}...")

    model = MLP().to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = opt_class(model.parameters(), lr=lr)

    losses = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            losses.append(loss_val)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    acc = evaluate(model)

    return {
        "name": name,
        "losses": losses,
        "accuracy": acc,
        "time": elapsed
    }


results = [
    run_experiment(SGD, "SGD", lr=0.01),
    run_experiment(Momentum, "Momentum", lr=0.01),
    run_experiment(Adam, "Adam", lr=0.001)
]

print("FINAL BENCHMARK RESULTS")

for r in results:
    print(f"{r['name']:<10} | Acc: {r['accuracy']:.2f}% | Time: {r['time']:.2f}s")


plt.figure(figsize=(10, 6))

for r in results:
    plt.plot(r['losses'], label=r['name'], alpha=0.8)

plt.yscale("log")
plt.xlabel("Iterations (Batches)")
plt.ylabel("Loss (Log Scale)")
plt.title("Optimizer Convergence Comparison")
plt.legend()
plt.grid(True, which="both", alpha=0.5)
plt.gca().set_facecolor("#000000")

plt.savefig("visuals/optimizer_benchmark.png")

print("\nBenchmark graph saved to visuals/optimizer_benchmark.png")