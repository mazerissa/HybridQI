# 🧠 HybridQI

### *Journal Entry*
I started with a simple question: *Can we teach a machine to see using only raw calculus?* Most people use "Black Box" libraries. I didn't. I built the brain (MLP) and the engines (Optimizers) from scratch to understand the physics of learning before we attempt to evolve it with Hybrid logic.

---

## **🧮 The Math**

### 1. The Mapping (Linear Algebra)
Every image is flattened into a vector $x \in \mathbb{R}^{784}$. I pass it through layers using:
$$y = \sigma(Wx + b)$$
Where $\sigma$ is **ReLU** ($\max(0, x)$). This "non-linearity" is what allows the model to learn complex shapes instead of just straight lines.

### 2. The Feedback (Cross-Entropy)
I measure "wrongness" by comparing our prediction ($\hat{y}$) to the truth ($y$):
$$\mathcal{L} = -\sum y \log(\hat{y})$$

### 3. The Engines (Optimizers)
I implemented three distinct ways to navigate the "Loss Landscape":

* **SGD :** Simple subtraction. $W = W - \eta \nabla L$.
* **Momentum:** Uses a velocity $v$ to blast through flat spots.
    $$v_t = \beta v_{t-1} + \nabla L \quad \rightarrow \quad W = W - \eta v_t$$
* **Adam:** Individually brakes or accelerates every single weight based on "vibration" ($v_t$) and "direction" ($m_t$).
    $$W = W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

## 🚀 Performance
Running on MNIST (Handwritten Digits):

| Optimizer | Accuracy | Convergence |
| :--- | :--- | :--- |
| **SGD** | 94.7% | Slow |
| **Momentum** | 97.4% | Fast |
| **Adam** | 97.5% | Instant |

---

## 📋 How to Use

### 1. Setup
```bash
python3 -m pip install -r requirements.txt
```

### 2. Run the Trainer

```bash
python3 -m src.training.trainer
```
### 3. Run the Benchmark
```bash
python3 -m experiments.benchmark
```
---

This project's still in development....
