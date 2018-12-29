import torch

x = torch.ones(2, 4, requires_grad=True)
w = torch.ones(4, 2, requires_grad=True)
print(x)
print(w)

z = 3 * torch.matmul(x, w)
print(z)
z.retain_grad()

y = z.norm()
print(y)

y.backward()
print(x.grad)
print(w.grad)

print(y.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)

print(z.grad)