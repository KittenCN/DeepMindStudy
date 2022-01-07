from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter 
# import torchvision
import dbhelper as db
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import os
from tqdm import tqdm

db_path = r'D:\workstation\GitHub\DeepMindStudy\data\Meteorological forecast\data\DB\database.db'
category_list = ['EVP', 'GST', 'PRE', 'PRS', 'RHU', 'SSD', 'TEM', 'WIN']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Use " + str(device))
ori_in_date = []
ori_out_data = [] 
_db = db.Connect(db_path)
pklfile = r"D:\workstation\GitHub\DeepMindStudy\data\Meteorological forecast\data\model\model.pkl"
writer = SummaryWriter(r'D:\workstation\GitHub\DeepMindStudy\Meteorological forecast\tf-logs')

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class rainfall:
    def __init__(self, id, allday_rainfall):
        self.id = id
        self.allday_rainfall = allday_rainfall

class metedata:
    def __init__(self, id, avg_temp, avg_humidity, avg_pressure, altitude):
        self.id = id
        self.avg_temp = avg_temp
        self.avg_humidity = avg_humidity
        self.avg_pressure = avg_pressure
        self.altitude = altitude

def checkdata(num):
    if num >= 30000:
        return False
    return True

if __name__ == "__main__":
    unvalidID = []
    rainfalllist = []
    metedatalist = []
    _table = _db.table("METE")
    _datas = _table.findAll()
    rowcnt = len(_datas)
    subbar = tqdm(total=rowcnt)
    for i, dt in enumerate(_datas):
        subbar.update(1)
        if (checkdata(int(dt['avg_pressure'])) == False or checkdata(int(dt['avg_humidity'])) == False or checkdata(int(dt['avg_temp'])) == False or checkdata(int(dt['allday_rainfall'])) == False or int(dt['avg_humidity']) <= 0) and i not in unvalidID:
            unvalidID.append(i)
        rainfalllist.append(rainfall(i, dt['avg_temp'])) 
        metedatalist.append(metedata(i, dt['avg_temp'], dt['avg_humidity'], dt['avg_pressure'], dt['altitude']))
    subbar.close()
    subbar = tqdm(total=len(rainfalllist))
    for i, dt in enumerate(rainfalllist):
        subbar.update(1)
        if i in unvalidID:
            continue
        tempdt = []
        if int(dt.allday_rainfall) > 0:
            tempdt.append(1)
        else:
            tempdt.append(0)
        ori_out_data.append(tempdt)
    subbar.close()
    subbar = tqdm(total=len(metedatalist))
    for i, dt in enumerate(metedatalist):
        subbar.update(1)
        if i in unvalidID:
            continue
        tempdt = []
        tempdt.append(float(int(dt.avg_temp)) / 10)
        tempdt.append(float(int(dt.avg_humidity)))
        tempdt.append(float(int(dt.avg_pressure)) / 100)
        ori_in_date.append(tempdt)    
    _db.close()
    subbar.close()
    # del ori_in_date[-1]
    # del ori_out_data[0]
    epochs = 3000
    lr = 0.01
    if os.path.exists(pklfile):
#         model = torch.load(pklfile).to(device)
        model = net().to(device) 
        model.load_state_dict(torch.load(pklfile))
        model.eval()
        print("load old model")
    else:
        model = net().to(device)
        print("creat new model")
    print("total data: ", rowcnt, "valid data: ", len(ori_in_date))
    in_data = torch.from_numpy(np.array(ori_in_date)).float()
    out_data = torch.from_numpy(np.array(ori_out_data)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss().to(device)
    dataset = TensorDataset(in_data, out_data)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    bar = tqdm(total=10, leave=False)
    for epoch in range(epochs):
        bar.update(1)
        subbar = tqdm(total=len(data_loader), leave=False)
        for inputs, targets in data_loader:
            subbar.update(1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            # grid = torchvision.utils.make_grid(inputs)
            # writer.add_image('inputs', grid, 0)
            # writer.add_graph(model, inputs)
            loss = loss_func(outputs, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar('Loss/train', float(loss), epoch)
        subbar.close()
        if (epoch + 1) % 10 == 0 and epoch != 0:
            bar.close()
            print('epoch:', epoch + 1, 'loss:', loss.item())
            bar = tqdm(total=10, leave=False)
        if (epoch + 1) % 100 == 0 and epoch != 0:
            torch.save(model.state_dict(), pklfile)
writer.close()
