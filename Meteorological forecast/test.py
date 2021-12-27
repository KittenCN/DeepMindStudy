from torch.utils.data.dataset import TensorDataset
import dbhelper as db
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import os
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()
db_path = 'Meteorological forecast/data/DB/database.db'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ori_in_date = []
ori_out_data = [] 
_db = db.Connect(db_path)
pklfile = "Meteorological forecast/model/model.pkl"

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

def checkdata(num):
    if num >= 30000:
        return False
    return True

def RP(t, h):
    a = 17.27
    b = 237.7
    y = ((a * t) / (b + t)) + math.log(h / 100)
    Td = (b * y) / (a - y)
    return Td

def show_data(x_data=[1, 2, 3], y_data=[1, 2, 3]):
    plt.scatter(x=x_data, y=y_data, color='k')
    plt.show()

if __name__ == '__main__':
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
        tempdt.append(RP(float(int(dt.avg_temp)) / 10, float(int(dt.avg_humidity))))
        ori_in_date.append(tempdt)    
    _db.close()
    model = torch.load(pklfile).to(device)
    model.eval()
    sumloss = 0
    plts = []
    for i, dt in enumerate(ori_in_date):
        x = [dt]
        x = torch.from_numpy(np.array(x)).float().to(device)
        plts.append([abs(dt[0] - dt[1]),ori_out_data[i][0]])
        y = model(x).item()
        if y >= 1:
            y = 1
        elif y <= 0:
            y = 0
        sumloss += abs(y - ori_out_data[i][0])
    print("avgloss = ", round(sumloss / (len(ori_in_date) * 100) * 100, 2), "%")