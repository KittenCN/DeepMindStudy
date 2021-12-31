from torch.utils.data.dataset import TensorDataset
import dbhelper as db
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import os
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from apex import amp
# from torch.utils.tensorboard import SummaryWriter
# import torchvision

db_path = r"Meteorological forecast/data/DB/database.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ori_in_date = []
ori_out_data = [] 
_db = db.Connect(db_path)
pklfile = r"Meteorological forecast/model/model.pkl"
# log_writer = SummaryWriter(r"/root/tf-logs/")

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

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

def RP(t, h):
    try:
        a = 17.27
        b = 237.7
        y = ((a * t) / (b + t)) + math.log(h / 100)
        Td = (b * y) / (a - y)
        return Td
    except BaseException:
        print(t, h)
        exit()

def checkdata(num):
    if num >= 30000:
        return False
    return True

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

if __name__ == "__main__":
    unvalidID = []
    rainfalllist = []
    metedatalist = []
    # strSQL = "select a.locationID, a.altitude, a.year, a.month, a.day, a.allday_rainfall from pre a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where 1 = 1 order by a.locationID, a.year, a.month, a.day"
    # _datas = _db.query(strSQL, True)
    # for i, dt in enumerate(_datas):
    #     if checkdata(int(dt[4])) == False and i not in unvalidID:
    #         unvalidID.append(i)
    #     rainfalllist.append(rainfall(i, dt[4])) 
    # strSQL = "select count(a.locationID) from prs a inner join rhu b inner join tem c inner join pre d on a.locationID = b.locationID and a.locationID = c.locationID and a.locationID = d.locationID and a.year = b.year and a.year = c.year and a.year = d.year and a.month = b.month and a.month = c.month and a.month = d.month and a.day = b.day and a.day = c.day and a.day = d.day where a.avg_pressure < 30000 and b.avg_humidity < 30000 and c.avg_temp < 3000 and d.allday_rainfall < 3000 and a.altitude < 90000 order by a.locationID, a.year, a.month, a.day"
    # # strSQL += " limit 1000"
    # _datas = _db.query(strSQL, True)
    # for dt in _datas:
    #     rowcnt = dt[0]
    #     break
    # strSQL = "select a.locationID, a.altitude, a.year, a.month, a.day, a.avg_pressure, b.avg_humidity, c.avg_temp, d.allday_rainfall from prs a inner join rhu b inner join tem c inner join pre d on a.locationID = b.locationID and a.locationID = c.locationID and a.locationID = d.locationID and a.year = b.year and a.year = c.year and a.year = d.year and a.month = b.month and a.month = c.month and a.month = d.month and a.day = b.day and a.day = c.day and a.day = d.day where a.avg_pressure < 30000 and b.avg_humidity < 30000 and c.avg_temp < 3000 and d.allday_rainfall < 3000 and a.altitude < 90000 order by a.locationID, a.year, a.month, a.day"
    # _datas = _db.query(strSQL, True)
    # strSQL += " limit 1000"
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
    # del ori_in_date[-1]
    # del ori_out_data[0]
    epochs = 100000
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
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    dataset = TensorDataset(in_data, out_data)
    # data_loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True)
    data_loader = DataLoaderX(dataset, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True)
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
            # log_writer.add_image('inputs', grid, 0)
            # log_writer.add_graph(model, inputs)
            loss = loss_func(outputs, targets.to(device))
            # log_writer.add_scalar('Loss/train', float(loss), epoch)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            optimizer.step()
        subbar.close()
        if (epoch + 1) % 10 == 0 and epoch != 0:
            bar.close()
            print('epoch:', epoch + 1, 'loss:', loss.item())
            bar = tqdm(total=10, leave=False)
        if (epoch + 1) % 100 == 0 and epoch != 0:
            torch.save(model.state_dict(), pklfile)
    # log_writer.close()

