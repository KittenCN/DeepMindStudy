from torch.utils.data.dataset import TensorDataset
import dbhelper as db
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import os

db_path = 'Meteorological forecast/data/DB/database.db'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ori_in_date = []
ori_out_data = [] 
_db = db.Connect(db_path)
pklfile = "Meteorological forecast/model/model.pkl"

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

def checkdata(num):
    if num >= 30000:
        return False
    return True

if __name__ == "__main__":
    strSQL1 = "select a.locationID, a.year, a.month, a.day, a.allday_rainfall from pre a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where a.locationID = 58362 order by a.year, a.month, a.day"
    _datas1 = _db.query(strSQL1, True)
    strSQL2 = "select a.locationID, a.year, a.month, a.day, a.avg_pressure, b.avg_humidity, c.avg_temp  from prs a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where a.locationID = 58362 order by a.year, a.month, a.day"
    _datas2 = _db.query(strSQL2, True)
    unvalidID = []
    for i, dt in enumerate(_datas1):
        if checkdata(int(dt[4])) == False and i not in unvalidID:
            unvalidID.append(i)
    for i, dt in enumerate(_datas2):
        if (checkdata(int(dt[4])) == False or checkdata(int(dt[5])) == False or checkdata(int(dt[6])) == False) and i not in unvalidID:
            unvalidID.append(i)
    trSQL1 = "select a.locationID, a.year, a.month, a.day, a.allday_rainfall from pre a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where a.locationID = 58362 order by a.year, a.month, a.day"
    _datas1 = _db.query(strSQL1, True)
    strSQL2 = "select a.locationID, a.year, a.month, a.day, a.avg_pressure, b.avg_humidity, c.avg_temp  from prs a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where a.locationID = 58362 order by a.year, a.month, a.day"
    _datas2 = _db.query(strSQL2, True)
    for i, dt in enumerate(_datas1):
        if i in unvalidID:
            continue
        tempdt = []
        if int(dt[4]) > 0:
            tempdt.append(1)
        else:
            tempdt.append(0)
        ori_out_data.append(tempdt)
    for i, dt in enumerate(_datas2):
        if i in unvalidID:
            continue
        tempdt = []
        # tempdt.append(float(checkdata(int(dt[4])) / 100))
        # tempdt.append(float(checkdata(int(dt[5])) / 100))
        tempdt.append(float(int(dt[6])) / 10)
        tempdt.append(RP(float(int(dt[6])) / 10, float(int(dt[5]))))
        ori_in_date.append(tempdt)    
    _db.close()
    # del ori_in_date[-1]
    # del ori_out_data[0]
    in_data = torch.from_numpy(np.array(ori_in_date)).float()
    out_data = torch.from_numpy(np.array(ori_out_data)).float()
    dataset = TensorDataset(in_data, out_data)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    epochs = 10000
    lr = 0.0001
    if os.path.exists(pklfile):
        model = torch.load(pklfile).to(device)
        model.eval()
    else:
        model = net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss().to(device)

    for epoch in range(epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('epoch:', epoch, 'loss:', loss.item())
        if epoch % 100 == 0:
            torch.save(model, pklfile)