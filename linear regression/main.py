import random
from typing import ForwardRef
from torch.utils.data.dataset import TensorDataset
import helper.aihelper as ai
import torch
import  torch.utils.data

use_gpu = torch.cuda.is_available()

class linear_net(torch.nn.Module):
    def __init__(self):
        super(linear_net, self).__init__()
        self.linear = torch.nn.Linear(in_features=1, out_features=1)
    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    a = 10
    b = 15
    c = 25
    num = 1000

    x = torch.randn(num).view(-1, 1)
    # hot_pixel = num / random.randint(0, 10)
    hot_pixel = 0
    y = a * x + b + hot_pixel
    dataset = TensorDataset(x, y)

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32)
    epochs = 10000
    lr = 1e-3
    model = linear_net()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    if(use_gpu):
        model = model.cuda()
        loss_func = loss_func.cuda()

    for epoch in range(epochs):
        lastepoch = -1
        for inputs, targets in data_loader:
            if(use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            if epoch % 10 == 0 and epoch != lastepoch:
                ai.show_data_cost(inputs, outputs, targets, loss, use_gpu)
                lastepoch = epoch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if epoch % 10 == 0:
            # print('epoch:', epoch, 'loss:', loss.item())
    
    print('a:', model.linear.weight.item())
    print('b:', model.linear.bias.item())
    print(list(model.parameters()))
    