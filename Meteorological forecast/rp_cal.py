import dbhelper as db
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

db_path = r"D:\\workstation\\GitHub\DeepMindStudy\data\\Meteorological forecast\\data\\DB\\database.db"
ori_in_date = []
ori_out_data = [] 
_db = db.Connect(db_path)

class metedata:
    def __init__(self, id, avg_temp, avg_humidity, avg_pressure, allday_rainfall):
        self.id = id
        self.avg_temp = float(int(avg_temp)) / 10
        self.avg_humidity = float(int(avg_humidity))
        self.avg_pressure = float(int(avg_pressure)) / 100
        self.allday_rainfall = allday_rainfall
        self.RP = RP(float(int(avg_temp)) / 10, float(int(avg_humidity)))
        self.dis = self.avg_temp - self.RP

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

unvalidID = []
rainfalllist = []
metedatalist = []
id = []
avg_tmp = []
avg_rp = []
avg_dis = []
avg_rain = []
_table = _db.table("METE")
_datas = _table.findAll()
rand_index = np.random.randint(0, len(_datas) - 101)
_datas = _datas[rand_index:rand_index + 100]
rowcnt = len(_datas)
subbar = tqdm(total=rowcnt)
for i, dt in enumerate(_datas):
    subbar.update(1)
    # if (checkdata(int(dt['avg_pressure'])) == False or checkdata(int(dt['avg_humidity'])) == False or checkdata(int(dt['avg_temp'])) == False or checkdata(int(dt['allday_rainfall'])) == False or int(dt['avg_humidity']) <= 0):
    #     continue
    id.append(str(dt['year']) + '-' + str(dt['month']) + '-' + str(dt['day']))
    avg_tmp.append(float(int(dt['avg_temp'])) / 10)
    if float(int(dt['allday_rainfall'])) > 0:
        avg_rain.append(1)
    else:
        avg_rain.append(0)
    avg_rp.append(RP(float(int(dt['avg_temp'])) / 10, float(int(dt['avg_humidity']))))
    avg_dis.append(float(int(dt['avg_temp'])) / 10 - RP(float(int(dt['avg_temp'])) / 10, float(int(dt['avg_humidity']))))
subbar.close()
ticker_spacing = id
ticker_spacing = 10
avg_tmp = np.array(avg_tmp)
avg_dis = np.array(avg_dis)
avg_rain = np.array(avg_rain)
avg_rp = np.array(avg_rp)
print(np.corrcoef(avg_dis, avg_rain))
fig, ax = plt.subplots(1, 1)
plt.plot(id, avg_tmp, label='avg_temp',color = 'y')
plt.plot(id, avg_rp, label='avg_rp',color = 'b')
# plt.plot(id, avg_dis,label='dew-point deficit',color = 'g')
plt.plot(id, avg_rain,label='rainfall rate',color = 'r')
ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))
plt.legend()
plt.show()