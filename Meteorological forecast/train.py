from torch.utils.data.dataset import TensorDataset
import dbhelper as db
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision

db_path = r"/root/MeteorologicalForecast/data/DB/database.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ori_in_date = []
ori_out_data = [] 
_db = db.Connect(db_path)
pklfile = r"/root/MeteorologicalForecast/model/model.pkl"
log_writer = SummaryWriter(r"/root/tf-logs/")

class net(nn.Module):
    def __init__(self) -> None:
        super(net, self).__init__()
        self.fc1 = nn.Linear(4, 1024)
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

class rainfall:
    def __init__(self, id, allday_rainfall):
        self.id = id
        self.allday_rainfall = allday_rainfall

class metedata:
    def __init__(self, id, avg_temp, avg_humidity, avg_pressure):
        self.id = id
        self.avg_temp = avg_temp
        self.avg_humidity = avg_humidity
        self.avg_pressure = avg_pressure

if __name__ == "__main__":
    unvalidID = []
    rainfalllist = []
    metedatalist = []
    strSQL = "select a.locationID, a.year, a.month, a.day, a.allday_rainfall from pre a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where a.locationID = 58362 order by a.year, a.month, a.day"
    _datas = _db.query(strSQL, True)
    for i, dt in enumerate(_datas):
        if checkdata(int(dt[4])) == False and i not in unvalidID:
            unvalidID.append(i)
        rainfalllist.append(rainfall(i, dt[4])) 
    strSQL = "select a.locationID, a.year, a.month, a.day, a.avg_pressure, b.avg_humidity, c.avg_temp  from prs a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where a.locationID = 58362 order by a.year, a.month, a.day"
    _datas = _db.query(strSQL, True)
    for i, dt in enumerate(_datas):
        if (checkdata(int(dt[4])) == False or checkdata(int(dt[5])) == False or checkdata(int(dt[6])) == False) and i not in unvalidID:
            unvalidID.append(i)
        metedatalist.append(metedata(i, dt[6], dt[5], dt[4]))
    for i, dt in enumerate(rainfalllist):
        if i in unvalidID:
            continue
        tempdt = []
        if int(dt.allday_rainfall) > 0:
            tempdt.append(1)
        else:
            tempdt.append(0)
        ori_out_data.append(tempdt)
    for i, dt in enumerate(metedatalist):
        if i in unvalidID:
            continue
        tempdt = []
        tempdt.append(float(int(dt.avg_temp)) / 10)
        tempdt.append(float(int(dt.avg_humidity)))
        tempdt.append(float(int(dt.avg_pressure)) / 100)
        tempdt.append(RP(float(int(dt.avg_temp)) / 10, float(int(dt.avg_humidity))))
        ori_in_date.append(tempdt)    
    _db.close()
    # del ori_in_date[-1]
    # del ori_out_data[0]
    epochs = 100000
    lr = 0.000001
    if os.path.exists(pklfile):
        model = torch.load(pklfile).to(device)
        model.eval()
        print("load old model")
    else:
        model = net().to(device)
        print("creat new model")
    in_data = torch.from_numpy(np.array(ori_in_date)).float().to(device)
    out_data = torch.from_numpy(np.array(ori_out_data)).float().to(device)
    dataset = TensorDataset(in_data, out_data)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss().to(device)
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            grid = torchvision.utils.make_grid(inputs)
            log_writer.add_image('inputs', grid, 0)
            log_writer.add_graph(model, inputs)
            loss = loss_func(outputs, targets.to(device))
            log_writer.add_scalar('Loss/train', float(loss), epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('epoch:', epoch, 'loss:', loss.item())
        if epoch % 1000 == 0 and epoch != 0:
            torch.save(model, pklfile)
    log_writer.close()
