from pickletools import optimize
import random
from turtle import forward
from matplotlib import use
from scipy import rand
from sklearn import datasets
from torch.utils.data.dataset import TensorDataset
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_gpu = torch.cuda.is_available()

class model(nn.Module):
    def __init__(self) -> None:
        super(model, self).__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=15)
        self.fc2 = nn.Linear(15, 30)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    k1 = 10
    k2 = 13
    k3 = 15
    b = 17

    epochs = 100
    x = []
    y = []
    for i in range(epochs):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        x3 = random.uniform(-1, 1)
        yd = k1 * x1 + k2 * x2 + k3 * x3 + b
        x.append([x1, x2, x3])
        y.append(yd)
    
    dataset = TensorDataset(torch.from_numpy(np.array([x]).float()),torch.from_numpy(np.array([y]).float()))
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32)
    
    lr = 1e-3
    net = model()
    opt = torch.optim.Adam(net.parameters(), lr)
    loss_f = nn.MSELoss()

    if(use_gpu):
        net = net.cuda()
        loss_f = loss_f.cuda()
    
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            if(use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = net(inputs)
            loss = loss_f(outputs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            print('epoch:', epoch, 'loss:', loss.item())