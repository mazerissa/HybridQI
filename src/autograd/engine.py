import numpy as np

def backward_engine(tensor):
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            if v._ctx:
                for parent in v._ctx.parents:
                    build_topo(parent)
            topo.append(v)
    
    build_topo(tensor)
    
    tensor.grad = np.ones_like(tensor.data)
    
    for node in reversed(topo):
        if node._ctx is None: continue
        
        grads = node._ctx.op.backward(node._ctx, node.grad)
        if not isinstance(grads, tuple): grads = (grads,)
        
        for parent, g in zip(node._ctx.parents, grads):
            if parent.requires_grad:
                parent.grad += g