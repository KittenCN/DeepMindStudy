from torch.functional import Tensor
from torch.utils.data.dataset import TensorDataset
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()

class linear_net(nn.Module):
    def __init__(self):
        super(linear_net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(Tensor.transpose(1, 1)).transpose(1, 1)
        return x
        

if __name__ == "__main__":
    a = 10
    b = 15
    c = 25
    num = 1000

    x = torch.randn(num).view(-1, 1)
    # hot_pixel = num / random.randint(0, 10)
    hot_pixel = 0
    y = a * x * x + b * x + c + hot_pixel
    dataset = TensorDataset(x, y)

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32)
    epochs = 50
    lr = 1e-3
    model = linear_net()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    if(use_gpu):
        model = model.cuda()
        loss_func = loss_func.cuda()

    listloss = []
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            if(use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        listloss.append(loss.item())
        if epoch % 10 == 0:
            print('epoch:', epoch, 'loss:', loss.item())
            ai.show_data_cost(inputs, outputs, targets, loss, use_gpu)
    
    # print('a:', model.linear.weight.item())
    # print('b:', model.linear.bias.item())
    print(list(model.parameters()))
    torch.save(model, "linear regression\model\model.pkl")
    x_data = []
    for i in range(len(listloss)):
        x_data.append(i)
    show_data(x_data, listloss)
    