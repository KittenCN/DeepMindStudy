from torch.utils.data.dataset import TensorDataset
import helper.aihelper as ai
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()

class nonlinear_net(nn.Module):
    def __init__(self):
        super(nonlinear_net, self).__init__()
        # self.linear = nn.Linear(in_features=1, out_features=1)
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(1, 10)
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100,50)
        self.fc5 = nn.Linear(50,1)

    def forward(self, x):
        # return self.linear(x)       
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = torch.cat((x1,x2),1)
        x = self.fc3(x3)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
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
    epochs = 500
    lr = 1e-3
    model = nonlinear_net()
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
    torch.save(model, "Nonlinear regression\model\model.pkl")
    x_data = []
    for i in range(len(listloss)):
        x_data.append(i)
    ai.show_data(x_data, listloss)
    