from torch.utils.data.dataset import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_gpu = torch.cuda.is_available()

class net(nn.Module):
    def __init__(self) -> None:
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)  
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    pressure = float(input("Please input pressure: "))
    humidity = float(input("Please input humidity: "))
    temp = float(input("Please input temperature: "))
    model = torch.load("Meteorological forecast/model/model.pkl")
    x = [[pressure, humidity, temp]]
    x = torch.from_numpy(np.array(x)).float()
    if use_gpu:
        x = x.cuda()
        model = model.cuda()
    y = model(x)
    print("rate = ", y.item())