import torch

x = torch.randn(100)
print(x)
x = x.view(-1,2)
print(x)