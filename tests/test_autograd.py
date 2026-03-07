from src.autograd.tensor import Tensor


x1 = Tensor(5.0, requires_grad=True)
y1 = x1.relu()
y1.backward()

print(f"ReLU(5.0) Grad: {x1.grad}")
x2 = Tensor(-5.0, requires_grad=True)
y2 = x2.relu()
y2.backward()
print(f"ReLU(-5.0) Grad: {x2.grad}")