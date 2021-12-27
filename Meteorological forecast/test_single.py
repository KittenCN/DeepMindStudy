from torch.utils.data.dataset import TensorDataset
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_gpu = torch.cuda.is_available()

class net(nn.Module):
    def __init__(self) -> None:
        super(net, self).__init__()
        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def RP(t, h):
    a = 17.27
    b = 237.7
    y = ((a * t) / (b + t)) + math.log(h / 100)
    Td = (b * y) / (a - y)
    return Td

if __name__ == '__main__':
    # pressure = float(input("Please input pressure: "))
    humidity = float(input("Please input humidity: "))
    temp = float(input("Please input temperature: "))
    pressure = float(input("Please input pressure: "))
    model = torch.load("Meteorological forecast/model/model.pkl")
    x = [[temp, humidity, pressure, RP(temp, humidity)]]
    x = torch.from_numpy(np.array(x)).float()
    if use_gpu:
        x = x.cuda()
        model = model.cuda()
    y = model(x)
    print("rate = ", y.item())