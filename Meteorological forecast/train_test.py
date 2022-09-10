import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

txt_list = glob.glob("data\Meteorological forecast\data\TXT\*.txt")
# print(txt_list)
list_pre = []
list_tem = []
list_rhu = []
list_prs = []
subbar = tqdm(total=len(txt_list))
for txtt in txt_list:
    subbar.update(1)
    # print(txtt)
    with open(txtt, 'r') as f:
        data = f.readlines()
        for line in data:           
            sp = line.split()
            # if sp[0] == "58362":
                # print(txtt)
                # print(sp)
            list = [sp[0], int(sp[4]), int(sp[5]), int(sp[6])]
            if txtt[59:62] == 'PRE':
                if sp[9] == '0':
                    list_pre.append(0)
                else:
                    list_pre.append(1)
            if txtt[59:62] == 'TEM':
                list_tem.append(int(sp[7]))
            if txtt[59:62] == 'RHU':
                list_rhu.append(int(sp[7]))
            if txtt[59:62] == 'PRS':
                list_prs.append(int(sp[7]))
subbar.close()
# print(list_tem)
# print(list_rhu)
# print(list_prs)
# print(list_pre)
pre_list_train = []
pre_list_test = []
rainfall_train = []
rainfall_test = []
# print(len(list_tem))
# print(len(list_rhu))
# print(len(list_prs))
# print(len(list_pre))
i = 0
end = len(list_tem)
list = []
while i < end:
    if int(list_tem[i]) <= 1000 and int(list_rhu[i]) <= 2000 and int(list_prs[i]) <= 20000:
        if i < 1450:
            tmp = [list_tem[i], list_rhu[i], list_prs[i]]
            pre_list_train.append(tmp)
            rainfall_train.append([list_pre[i]])
        else:
            tmp = [list_tem[i], list_rhu[i], list_prs[i]]
            pre_list_test.append(tmp)
            rainfall_test.append([list_pre[i]])
    i += 1
    # if int(list_tem[i]) > 6000 or int(list_rhu[i]) > 2000 or int(list_prs[i]) > 20000:
    #     # print('wrong number i=', i)
    #     end = end - 1
    #     i = i + 1
    #     continue
    # list.append(float(list_tem[i]))
    # list.append(float(list_rhu[i][4]))
    # list.append(float(list_prs[i][4]))
    # if i < 1450:
    #     pre_list_train.append(list)
    #     tmp = []
    #     tmp.append(float(list_pre[i][4]))
    #     rainfall_train.append(tmp)
    # else:
    #     pre_list_test.append(list)
    #     tmp = []
    #     tmp.append(float(list_pre[i][4]))
    #     rainfall_test.append(tmp)
    # # print('okay number i=', i)
    # i = i + 1
# print(pre_list_train)
# print(pre_list_test)
# pltrain=[]
# pltrain.append(pre_list_train)
# pltest=[]
# pltest.append(pre_list_test)
# rtrain=[]
# rtrain.append(rainfall_train)
# rtest=[]
# rtest.append(rainfall_test)
pre_train = torch.from_numpy(np.array(pre_list_train)).float()
pre_test = torch.from_numpy(np.array(pre_list_test)).float()
rain_train = torch.from_numpy(np.array(rainfall_train)).float()
rain_test = torch.from_numpy(np.array(rainfall_test)).float()
# print("pre_train", pre_train)
# print("pre_test", pre_test)
train_dataset = torch.utils.data.TensorDataset(pre_train, rain_train)
test_dataset = torch.utils.data.TensorDataset(pre_test, rain_test)

epochs = 100
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class weather(nn.Module):
    def __init__(self):
        super(weather, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


if __name__ == "__main__":
    net = weather()
    loss_function = nn.MSELoss()
    lr = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            outputs = net(inputs)
            optimizer.zero_grad()
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            for inputs, targets in test_loader:
                outputs = net(inputs)
                loss = loss_function(outputs, targets)
                print('epoch:', epoch + 1, 'loss:', loss.item())
