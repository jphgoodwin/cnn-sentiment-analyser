import torch
import torch.nn.functional as fn
import pdb


x_in = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x_in)

k_size = 2
kernel = torch.randn(k_size)
print(kernel)

x_expanded = []
for i in range(0, x_in.size()[0] - 1):
    x_expanded.append([x_in[i].data, x_in[i+1].data])

kernel = kernel.reshape(k_size, 1)
x_expanded = torch.tensor(x_expanded)
print(x_expanded)
print(kernel)

result = torch.matmul(x_expanded, kernel)
