# 🧠 HybridQI

### *Journal Entry*
I started with a simple question: *Can we teach a machine to see using only raw calculus?* Most people use "Black Box" libraries. I didn't. After building the MLP and the Optimizers, I have now successfully built my own **Autograd Engine**. I don't rely on pre-made backpropagation.

---

## **🧮 The Math**

### 1. Linear Algebra
Every image is flattened into a vector $x \in \mathbb{R}^{784}$. I pass it through layers using:
$$y = \sigma(Wx + b)$$
Where $\sigma$ is **ReLU** ($\max(0, x)$). This "non-linearity" is what allows the model to learn complex shapes instead of just straight lines.

### 2. Cross-Entropy
I measure "wrongness" by comparing our prediction ($\hat{y}$) to the truth ($y$):
$$\mathcal{L} = -\sum y \log(\hat{y})$$

### 3. Optimizers
I implemented three distinct ways to navigate the "Loss Landscape":
* **SGD:** Simple subtraction. $W = W - \eta \nabla L$.
* **Momentum:** Uses a velocity $v$ to blast through flat spots.
    $$v_t = \beta v_{t-1} + \nabla L \quad \rightarrow \quad W = W - \eta v_t$$
* **Adam:** Individually brakes or accelerates every weight based on "vibration" ($v_t$) and "direction" ($m_t$).
    $$W = W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 4. Autograd
I have moved from static math to a **Computational Graph**. Every Tensor now stores its parents and the operation that created it.
* **Topological Sort:** Gradients are calculated in reverse order.
* **The Chain Rule:** Every operation now has a manual derivative definition.

---

## 🚀 Performance

| Optimizer | Accuracy |
| :--- | :--- |
| **SGD** | 94.7% |
| **Momentum** | 97.4% |
| **Adam** | 97.5% |

---

## 📋 How to Use

### 1. Setup
```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.
```

### 2. Test the Autograd Engine

```bash
python3 tests/test_autograd.py
```
### 3. Run the Benchmark
```bash
python3 -m experiments.benchmark
```
---

This project's still in development....
