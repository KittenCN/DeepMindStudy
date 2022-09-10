# 导入需要的模块
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用来绘图的，封装了matplot
# 要注意的是一旦导入了seaborn，
# matplotlib的默认作图风格就会被覆盖成seaborn的格式
import seaborn as sns       

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline  
# 为了在jupyter notebook里作图，需要用到这个命令

data_train = pd.read_csv("data/house prices/train.csv")

sns.distplot(data_train['SalePrice'])
plt.show()
