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
from tqdm import tqdm

use_gpu = torch.cuda.is_available()
db_path = r"Meteorological forecast\data\DB\database.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ori_in_date = []
ori_out_data = [] 
_db = db.Connect(db_path)
pklfile = r"/root/MeteorologicalForecast/model/model.pkl"

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

class net(nn.Module):
    def __init__(self) -> None:
        super(net, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
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
        tempdt.append(float(int(dt.altitude)) / 10)
        tempdt.append(RP(float(int(dt.avg_temp)) / 10, float(int(dt.avg_humidity))))
        ori_in_date.append(tempdt)    
    _db.close()
    subbar.close()
    model = net().to(device) 
    model.load_state_dict(torch.load(pklfile))
    model.eval()
    sumloss = 0
    plts = []
    subbar = tqdm(total=len(ori_in_date))
    for i, dt in enumerate(ori_in_date):
        subbar.update(1)
        x = [dt]
        x = torch.from_numpy(np.array(x)).float().to(device)
        plts.append([abs(dt[0] - dt[1]),ori_out_data[i][0]])
        y = model(x).item()
        if y >= 1:
            y = 1
        elif y <= 0:
            y = 0
        sumloss += abs(y - ori_out_data[i][0])
    subbar.close()
    print("avgloss = ", round(sumloss / (len(ori_in_date) * 100) * 100, 2), "%")