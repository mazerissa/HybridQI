import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Type

from src.models.mlp import MLP
from src.optimizers.sgd import SGD
from src.optimizers.momentum import Momentum
from src.optimizers.adam import Adam
from src.training.trainer import Trainer


def get_data_loaders(
    dataset_name: str = "MNIST",
    batch_size: int = 64,
    test_batch_size: int = 1000,
    data_dir: str = "./data"
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(
    model_class: Type[nn.Module],
    optimizer_class: Type,
    optimizer_config: Dict,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    device: str = "cpu",
    verbose: bool = True
) -> Dict:

    model = model_class().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_config)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs
    )

    start_time = time.time()
    history = trainer.train(train_loader, test_loader, verbose=verbose)
    elapsed_time = time.time() - start_time

    final_accuracy = history["test_accuracies"][-1] if history["test_accuracies"] else 0.0
    final_loss = history["train_losses"][-1] if history["train_losses"] else 0.0

    return {
        "model": model,
        "optimizer_name": optimizer_class.__name__,
        "history": history,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "training_time": elapsed_time,
        "total_params": sum(p.numel() for p in model.parameters())
    }


def compute_metrics(results: List[Dict]) -> Dict:

    metrics = {}
    for result in results:
        opt_name = result["optimizer_name"]
        metrics[opt_name] = {
            "accuracy": result["final_accuracy"],
            "loss": result["final_loss"],
            "time": result["training_time"],
            "params": result["total_params"]
        }
    return metrics


def print_results(results: List[Dict], metrics: Dict) -> None:

    print(f"\n{'Optimizer':<15} | {'Accuracy':<12} | {'Loss':<12} | {'Time (s)':<12}")

    for result in results:
        opt_name = result["optimizer_name"]
        metric = metrics[opt_name]
        print(
            f"{opt_name:<15} | {metric['accuracy']:>10.2f}% | "
            f"{metric['loss']:>10.4f} | {metric['time']:>10.2f}s"
        )


def print_detailed_results(results: List[Dict], metrics: Dict) -> None:

    for result in results:
        opt_name = result["optimizer_name"]
        metric = metrics[opt_name]
        history = result["history"]

        print(f"\n{opt_name} Optimizer:")
        print(f"  Final Test Accuracy: {metric['accuracy']:.2f}%")
        print(f"  Final Training Loss: {metric['loss']:.4f}")
        print(f"  Training Time: {metric['time']:.2f}s")
        print(f"  Total Parameters: {metric['params']:,}")

        if history["test_accuracies"]:
            print(f"  Min Accuracy: {min(history['test_accuracies']):.2f}%")
            print(f"  Max Accuracy: {max(history['test_accuracies']):.2f}%")
            improvements = sum(
                1 for i in range(1, len(history['test_accuracies']))
                if history['test_accuracies'][i] > history['test_accuracies'][i-1]
            )
            print(f"  Epochs with improvement: {improvements}")


def plot_results(
    results: List[Dict],
    save_dir: str = "visuals",
    save_plots: bool = True
) -> None:

    plt.style.use("dark_background")

    if save_dir and save_plots:
        Path(save_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for result in results:
        accuracies = result["history"]["test_accuracies"]
        if accuracies:
            ax.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label=result["optimizer_name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for result in results:
        batch_losses = result["history"]["batch_losses"]
        if batch_losses:
            step = max(1, len(batch_losses) // 200)
            ax.plot(range(0, len(batch_losses), step), batch_losses[::step], alpha=0.7, label=result["optimizer_name"])
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss per Batch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    optimizers = [r["optimizer_name"] for r in results]
    accuracies = [r["final_accuracy"] for r in results]
    colors = plt.cm.viridis(range(len(optimizers)))
    bars = ax.bar(optimizers, accuracies, color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final Test Accuracy Comparison")
    ax.set_ylim(0, 105)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.2f}%', ha='center', va='bottom')

    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    times = [r["training_time"] for r in results]
    bars = ax.bar(optimizers, times, color=colors)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time Comparison")

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{t:.2f}s', ha='center', va='bottom')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_plots:
        plot_path = Path(save_dir) / "training_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved at: {plot_path}")

    plt.show()


def run_tests(
    dataset: str = "MNIST",
    epochs: int = 5,
    batch_size: int = 64,
    test_batch_size: int = 1000,
    device: Optional[str] = None,
    optimizers_to_test: Optional[List[str]] = None,
    plot_results_flag: bool = True,
    save_plots: bool = True
) -> Dict:

    print("Starting optimizer benchmark...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[INFO] Using device: {device.upper()}")

    print(f"[INFO] Loading {dataset} dataset...")
    train_loader, test_loader = get_data_loaders(
        dataset_name=dataset,
        batch_size=batch_size,
        test_batch_size=test_batch_size
    )

    print(f"[INFO]   Training samples: {len(train_loader.dataset)}")
    print(f"[INFO]   Test samples: {len(test_loader.dataset)}")

    optimizer_configs = {
        "SGD": (SGD, {"lr": 0.01}),
        "Momentum": (Momentum, {"lr": 0.01, "beta": 0.9}),
        "Adam": (Adam, {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8})
    }

    if optimizers_to_test is None:
        optimizers_to_test = list(optimizer_configs.keys())

    print(f"\n[INFO] Training models with {len(optimizers_to_test)} optimizer(s)...\n")

    results = []

    for opt_name in optimizers_to_test:

        if opt_name not in optimizer_configs:
            print(f"[WARNING] {opt_name} not in available optimizers. Skipping.")
            continue

        opt_class, opt_config = optimizer_configs[opt_name]

        print(f"\nRunning with optimizer: {opt_name}")

        result = train_model(
            model_class=MLP,
            optimizer_class=opt_class,
            optimizer_config=opt_config,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            device=device,
            verbose=True
        )

        results.append(result)

    print("\nAll optimizers finished successfully.")

    metrics = compute_metrics(results)

    print_results(results, metrics)
    print_detailed_results(results, metrics)

    if plot_results_flag and len(results) > 0:
        print("\n[INFO] Generating plots...")
        plot_results(results, save_plots=save_plots)

    return {
        "results": results,
        "metrics": metrics,
        "device": device,
        "dataset": dataset,
        "epochs": epochs
    }


if __name__ == "__main__":

    all_results = run_tests(
        dataset="FashionMNIST",
        epochs=5,
        batch_size=64,
        test_batch_size=1000,
        optimizers_to_test=["SGD", "Momentum", "Adam"],
        plot_results_flag=True,
        save_plots=True
    )

    print("\n[INFO] Testing complete!")