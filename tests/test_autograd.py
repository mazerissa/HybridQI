from src.autograd.tensor import Tensor

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = a * b
d = c.relu()
e = d + a

print(f"Forward Result (e): {e.data}")
e.backward()
print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")