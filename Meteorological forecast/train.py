from torch.utils.data.dataset import TensorDataset
import dbhelper as db
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

db_path = 'Meteorological forecast/data/DB/database.db'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ori_in_date = []
ori_out_data = [] 
_db = db.Connect(db_path)

class net(nn.Module):
    def __init__(self) -> None:
        super(net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv1d(64, 128, 3, 1, 1)
        self.pool1 = nn.MaxPool1d(2, 2)
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

def checkdata(num):
    if num >= 30000:
        num = 0
    return num

if __name__ == "__main__":
    strSQL = "select a.locationID, a.year, a.month, a.day, a.allday_rainfall from pre a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where a.locationID = 58362 order by a.year, a.month, a.day"
    _datas = _db.query(strSQL, True)
    for i, dt in enumerate(_datas):
        tempdt = []
        tempdt.append(float(checkdata(int(dt[4])) / 10))
        ori_out_data.append(tempdt)
    strSQL = "select a.locationID, a.year, a.month, a.day, a.avg_pressure, b.avg_humidity, c.avg_temp  from prs a inner join rhu b inner join tem c on a.locationID = b.locationID and a.locationID = c.locationID and a.year = b.year and a.year = c.year and a.month = b.month and a.month = c.month and a.day = b.day and a.day = c.day where a.locationID = 58362 order by a.year, a.month, a.day"
    _datas = _db.query(strSQL, True)
    for i, dt in enumerate(_datas):
        tempdt = []
        tempdt.append(float(checkdata(int(dt[4])) / 100))
        tempdt.append(float(checkdata(int(dt[5])) / 100))
        tempdt.append(float(checkdata(int(dt[6])) / 10))
        ori_in_date.append(tempdt)
    _db.close()
    del ori_in_date[-1]
    del ori_out_data[0]
    in_data = torch.from_numpy(np.array(ori_in_date)).float().view(-1,3)
    out_data = torch.from_numpy(np.array(ori_out_data)).float().view(-1,1)
    dataset = TensorDataset(in_data, out_data)
    data_loader = DataLoader(dataset, batch_size=8)
    epochs = 1000
    lr = 0.1
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
    torch.save(model, "Meteorological forecast/model/model.pkl")