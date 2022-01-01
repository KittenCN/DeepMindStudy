import os
import re
from tqdm import tqdm
import dbhelper as db

txt_path = 'Meteorological forecast/data/TXT'
db_path = 'Meteorological forecast/data/DB/database.db'
category_list = ['EVP', 'GST', 'PRE', 'PRS', 'RHU', 'SSD', 'TEM', 'WIN']
files = os.listdir(txt_path)
datas = [[] for i in range(8)]

EVP = {'locationID':0, 'longitude':0, 'latitude':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'highest_evaporation':0, 'lowest_evaporation':0}
GST ={'locationID':0, 'longituda':0, 'latitude':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'avg_temp':0, 'hightlest_temp':0, 'lowest_temp':0}
PRE = {'locationID':0, 'longitude':0, 'latitude':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'night_rainfall':0, 'day_rainfall':0, 'allday_rainfall':0}
PRS = {'locationID':0, 'longitude':0, 'latitude':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'avg_pressure':0, 'hightlest_pressure':0, 'lowest_pressure':0}
RHU = {'locationID':0, 'longitude':0, 'latitude':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'avg_humidity':0, 'lowest_humidity':0}
SSD = {'locationID':0, 'longitude':0, 'latitude':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'sunshine':0}
TEM = {'locationID':0, 'longitude':0, 'latitude':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'avg_temp':0, 'hightlest_temp':0, 'lowest_temp':0}
WIN = {'locationID':0, 'longitude':0, 'latitude':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'avg_velocity':0, 'high_velocity':0, 'high_direction':0, 'highest_velocity':0, 'highest_direction':0}
METE = {'locationID':0, 'altitude':0, 'year':0, 'month':0, 'day':0, 'avg_pressure':0, 'avg_humidity':0, 'avg_temp':0, 'allday_rainfall':0}
category_group = [EVP, GST, PRE, PRS, RHU, SSD, TEM, WIN]

def get_data(origin, filename, index):
    with open(os.path.join(txt_path, filename), 'r') as f:
        lines = f.readlines()
        subpbar = tqdm(total=len(lines), leave=False)
        for line in lines:
            subpbar.update(1)
            line = line.strip()
            line = re.sub(r"\s+", " ", line)
            seges = line.split(' ')
            dict = origin.copy()
            for i, seg in enumerate(dict):
                dict[seg] = float(seges[i])
            datas[index].append(dict)
        subpbar.close()

def TransNewData():
    pbar = tqdm(total=len(files))
    for file in files:
        pbar.update(1)
        if file[21:24] in category_list:
            get_data(category_group[category_list.index(file[21:24])], file, category_list.index(file[21:24]))
    pbar.close()
    pbar = tqdm(total=len(datas))
    for i, dt in enumerate(datas):
        pbar.update(1)
        if dt.__len__() > 0:
            _con = db.Connect(db_path)
            _con.table(category_list[i]).data(dt).add()
            _con.close()
    pbar.close()

def CalulateMereForeData():
    _db = db.Connect(db_path)
    strSQL = "select a.locationID, a.altitude, a.year, a.month, a.day, a.avg_pressure, b.avg_humidity, c.avg_temp, d.allday_rainfall from prs a inner join rhu b inner join tem c inner join pre d on a.locationID = b.locationID and a.locationID = c.locationID and a.locationID = d.locationID and a.year = b.year and a.year = c.year and a.year = d.year and a.month = b.month and a.month = c.month and a.month = d.month and a.day = b.day and a.day = c.day and a.day = d.day where a.avg_pressure < 30000 and b.avg_humidity < 30000 and c.avg_temp < 3000 and d.allday_rainfall < 3000 and a.altitude < 90000 order by a.locationID, a.year, a.month, a.day"
    # strSQL += " limit 1000"
    _datas = _db.query(strSQL, True)
    data = []
    for i, dt in enumerate(_datas):
        dict = METE.copy()
        for j, seg in enumerate(dict):
            dict[seg] = dt[j]
        data.append(dict)
    _db.table('METE').data(data).add()
    _db.close()

if __name__ == "__main__":
    choice = input("enter your choice:")
    if choice == '1':
        TransNewData()
    elif choice == '2':
        CalulateMereForeData()
    else:
        quit()