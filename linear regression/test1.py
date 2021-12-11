import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.utils.data import DataLoader

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.lin = nn.Linear(1,1)
    
    def forward(self, x):
        return self.lin(x)

x = torch.randn(256).view(-1,1)
y = 3*x + 1

dataset = TensorDataset(x, y)

dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

epochs = 250
lr = 1e-3
model = Linear()
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_function = nn.MSELoss()

for epoch in range(epochs):
    for inputs, targets in dataloader:
        predict = model(inputs)
        loss = loss_function(predict, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(list(model.parameters()))