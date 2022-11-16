from matplotlib import pyplot as plt
import pandas as pd
# import pandas_profiling as ppf
from pandas_profiling import ProfileReport
import seaborn as sns
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder

class label_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        lab=LabelEncoder()
        X["YearBuilt"]=lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"]=lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"]=lab.fit_transform(X["GarageYrBlt"])
        return X

class skew_dumies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew
    def fit(self, X):
        return self
    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=['object'])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) > self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X

# import data
df_train = pd.read_csv(r'D:\workstation\GitHub\DeepMindStudy\data\house prices\train.csv')
df_test = pd.read_csv(r'D:\workstation\GitHub\DeepMindStudy\data\house prices\test.csv')

# data clean
df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<200000)].index, inplace=True)
df_train.drop(df_train[(df_train['TotalBsmtSF']>3000) & (df_train['SalePrice']<400000)].index, inplace=True)
df_train.drop(df_train[(df_train['OverallQual']>9) & (df_train['SalePrice']<200000)].index, inplace=True)
df_train.drop(df_train[(df_train['YearBuilt']<1900) & (df_train['SalePrice']>400000)].index, inplace=True)
df_train.drop(df_train[(df_train['YearBuilt']>1980) & (df_train['SalePrice']>700000)].index, inplace=True)

df_full = pd.concat([df_train, df_test], ignore_index=True)
df_full.drop(['Id'], axis=1, inplace=True)
# print(df_full.shape)

df_sum = df_full.isnull().sum()
# df_sum[df_sum>0].sort_values(ascending=False)
# print(df_sum[df_sum>0].sort_values(ascending=False))
df_full["LotAreaCut"] = pd.qcut(df_full["LotArea"], 10)
# print(df_full.groupby(['LotAreaCut'])['LotFrontage'].agg(['mean', 'median', 'count']))
df_full["LotFrontage"] = df_full.groupby(["LotAreaCut", "Neighborhood"])["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df_full["LotFrontage"] = df_full.groupby(["LotAreaCut"])["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# print(df_full.groupby(['LotAreaCut'])['LotFrontage'].agg(['mean', 'median', 'count']))

cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    df_full[col].fillna(0, inplace=True)

cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    df_full[col].fillna("None", inplace=True)

cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
for col in cols2:
    df_full[col].fillna(df_full[col].mode()[0], inplace=True)

# print(df_full.isnull().sum()[df_full.isnull().sum()>0])

NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    df_full[col] = df_full[col].astype(str)

def map_values():
    df_full["oMSSubClass"] = df_full.MSSubClass.map({'180':1, 
                                        '30':2, '45':2, 
                                        '190':3, '50':3, '90':3, 
                                        '85':4, '40':4, '160':4, 
                                        '70':5, '20':5, '75':5, '80':5, '150':5,
                                        '120': 6, '60':6})
    
    df_full["oMSZoning"] = df_full.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4})
    
    df_full["oNeighborhood"] = df_full.Neighborhood.map({'MeadowV':1,
                                               'IDOTRR':2, 'BrDale':2,
                                               'OldTown':3, 'Edwards':3, 'BrkSide':3,
                                               'Sawyer':4, 'Blueste':4, 'SWISU':4, 'NAmes':4,
                                               'NPkVill':5, 'Mitchel':5,
                                               'SawyerW':6, 'Gilbert':6, 'NWAmes':6,
                                               'Blmngtn':7, 'CollgCr':7, 'ClearCr':7, 'Crawfor':7,
                                               'Veenker':8, 'Somerst':8, 'Timber':8,
                                               'StoneBr':9,
                                               'NoRidge':10, 'NridgHt':10})
    
    df_full["oCondition1"] = df_full.Condition1.map({'Artery':1,
                                           'Feedr':2, 'RRAe':2,
                                           'Norm':3, 'RRAn':3,
                                           'PosN':4, 'RRNe':4,
                                           'PosA':5 ,'RRNn':5})
    
    df_full["oBldgType"] = df_full.BldgType.map({'2fmCon':1, 'Duplex':1, 'Twnhs':1, '1Fam':2, 'TwnhsE':2})
    
    df_full["oHouseStyle"] = df_full.HouseStyle.map({'1.5Unf':1, 
                                           '1.5Fin':2, '2.5Unf':2, 'SFoyer':2, 
                                           '1Story':3, 'SLvl':3,
                                           '2Story':4, '2.5Fin':4})
    
    df_full["oExterior1st"] = df_full.Exterior1st.map({'BrkComm':1,
                                             'AsphShn':2, 'CBlock':2, 'AsbShng':2,
                                             'WdShing':3, 'Wd Sdng':3, 'MetalSd':3, 'Stucco':3, 'HdBoard':3,
                                             'BrkFace':4, 'Plywood':4,
                                             'VinylSd':5,
                                             'CemntBd':6,
                                             'Stone':7, 'ImStucc':7})
    
    df_full["oMasVnrType"] = df_full.MasVnrType.map({'BrkCmn':1, 'None':1, 'BrkFace':2, 'Stone':3})
    
    df_full["oExterQual"] = df_full.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    df_full["oFoundation"] = df_full.Foundation.map({'Slab':1, 
                                           'BrkTil':2, 'CBlock':2, 'Stone':2,
                                           'Wood':3, 'PConc':4})
    
    df_full["oBsmtQual"] = df_full.BsmtQual.map({'Fa':2, 'None':1, 'TA':3, 'Gd':4, 'Ex':5})
    
    df_full["oBsmtExposure"] = df_full.BsmtExposure.map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})
    
    df_full["oHeating"] = df_full.Heating.map({'Floor':1, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':5})
    
    df_full["oHeatingQC"] = df_full.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    df_full["oKitchenQual"] = df_full.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    df_full["oFunctional"] = df_full.Functional.map({'Maj2':1, 'Maj1':2, 'Min1':2, 'Min2':2, 'Mod':2, 'Sev':2, 'Typ':3})
    
    df_full["oFireplaceQu"] = df_full.FireplaceQu.map({'None':1, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    df_full["oGarageType"] = df_full.GarageType.map({'CarPort':1, 'None':1,
                                           'Detchd':2,
                                           '2Types':3, 'Basment':3,
                                           'Attchd':4, 'BuiltIn':5})
    
    df_full["oGarageFinish"] = df_full.GarageFinish.map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})
    
    df_full["oPavedDrive"] = df_full.PavedDrive.map({'N':1, 'P':2, 'Y':3})
    
    df_full["oSaleType"] = df_full.SaleType.map({'COD':1, 'ConLD':1, 'ConLI':1, 'ConLw':1, 'Oth':1, 'WD':1,
                                       'CWD':2, 'Con':3, 'New':3})
    
    df_full["oSaleCondition"] = df_full.SaleCondition.map({'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4})

    return "Done!"

print(map_values())
df_full.drop("LotAreaCut", axis=1, inplace=True)
df_full.drop(["SalePrice"],axis=1, inplace=True)

pipe = Pipeline([
    ("labenc", label_encoder()),
    ("skew_dummies", skew_dumies(skew=1))
    ])
df_full2 = df_full.copy()
data_pipe = pipe.fit_transform(df_full2)
print(data_pipe.shape)

# print(df_full.groupby(['MSSubClass'])['SalePrice'].agg(['mean', 'median', 'count']))

# data analysis
# plt.figure(figsize=(18,12))
# plt.subplot(2,2,1)
# plt.scatter(x=df_train.YearBuilt, y=df_train.SalePrice,color='b')
# plt.xlabel("YearBuilt", fontsize=12)
# plt.ylabel("SalePrice", fontsize=12)

# plt.subplot(2,2,2)
# plt.scatter(x=df_train.GrLivArea, y=df_train.SalePrice,color='r')
# plt.xlabel("GrLivArea", fontsize=12)
# plt.ylabel("SalePrice", fontsize=12)

# plt.subplot(2,2,3)
# plt.scatter(x=df_train.TotalBsmtSF, y=df_train.SalePrice,color='g')
# plt.xlabel("TotalBsmtSF", fontsize=12)
# plt.ylabel("SalePrice", fontsize=12)

# plt.subplot(2,2,4)
# plt.scatter(x=df_train.OverallQual, y=df_train.SalePrice,color='y')
# plt.xlabel("OverallQual", fontsize=12)
# plt.ylabel("SalePrice", fontsize=12)

# plt.show()
