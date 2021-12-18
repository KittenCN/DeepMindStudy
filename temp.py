import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()

class linear_net(nn.Module):
    def __init__(self):
        super(linear_net, self).__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=2)
        self.fc2 = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x  

def show_data_cost(x_data=[1, 2, 3], y_data=[1, 2, 3], prediction_data=[1, 2, 3], loss=0, use_gpu=1):
    plt.cla()
    if use_gpu:
        x_data = x_data.cpu().detach().numpy()
        y_data = y_data.cpu().detach().numpy()
        prediction_data = prediction_data.cpu().detach().numpy()
        loss = loss.cpu().detach().numpy()
    else:
        x_data = x_data.detach().numpy()
        y_data = y_data.detach().numpy()
        prediction_data = prediction_data.detach().numpy()
        loss = loss.detach().numpy()
    plt.scatter(x_data, y_data, c='r')
    plt.scatter(x_data, prediction_data, c='b')
    plt.text(0,0,'Loss=%.4f'%loss, fontdict={'size':20,'color':'red'})
    plt.pause(0.1)#画的图只存在0.1秒

if __name__ == "__main__":
    w = 10
    w2 = 20
    b = 15
    num = 1000

    x = torch.randn(num).view(-1, 1)
    y = w * x * x + w2 * x + b
    dataset = TensorDataset(x, y)

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32)
    epochs = 1000
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
    