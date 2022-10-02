import torch.onnx 
import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

pklfile = r'D:\\workstation\\GitHub\DeepMindStudy\\linear regression\\model\\model.pkl'

class linear_net(nn.Module):
    def __init__(self):
        super(linear_net, self).__init__()
        # self.linear = nn.Linear(in_features=1, out_features=1) # 全连接层  
        self.fc1 = nn.Linear(in_features=1, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        # return self.linear(x) 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


model_path = pklfile                           # 模型参数路径  
dummy_input = torch.randn(1000).view(-1, 1)  # 先随机一个模型输入的数据
model = linear_net()                          # 定义模型结构，此处是我自己设计的模型
checkpoing = torch.load(pklfile, 'cpu')  # 导入模型参数
model.load_state_dict(checkpoing)           # 将模型参数赋予自定义的模型
torch.onnx.export(model, dummy_input, "model_best.onnx",verbose=True) # 将模型保存成.onnx格式