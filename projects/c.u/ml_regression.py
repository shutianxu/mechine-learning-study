# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
购车重要因素分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = ['device_number' ,'cust_sex' ,'years_old' ,'is_birth' ,'is_marry' ,'is_job' ,'car_owner' ,'area_price' ,'distance' ,'daohang_cnt' ,'daohang_dura' ,'didi_cnt' ,'didi_dura' ,'autohome_cnt' ,'autohome_dura' ,'gongxiangdanche_cnt' ,'gongxiangdanche_dura' ,'autohome_cnt_201712' ,'autohome_dura_201712' ,'autohome_cnt_201711' ,'autohome_dura_201711' ,'gongxiangqiche_cnt' ,'gongxiangqiche_dura' ,'foot_distance' ,'bike_distance' ,'car_distance' ,'vis_dealer_cnt' ,'cnxiandai_dealer_cnt' ,'gas_vis_cnt']
data = pd.read_csv("C:/Users/1707500/Desktop/cusc_data.csv", sep = '\t',encoding = 'gb2312',names = header)

# =============================================================================
# print(df)
# =============================================================================

orig_columns = data.columns
drop_columns = []
for col in orig_columns:
    col_series = data[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
data = data.drop(drop_columns, axis=1)
print(drop_columns)


data = data.drop(["device_number","didi_cnt","didi_dura","autohome_cnt_201711","autohome_dura_201711","gongxiangqiche_cnt","gongxiangqiche_dura","gas_vis_cnt","cnxiandai_dealer_cnt","daohang_cnt","gongxiangdanche_cnt","autohome_cnt_201712"],axis=1)

data = data.drop(["autohome_cnt","autohome_dura"],axis=1)




# =============================================================================
# data = data.dropna(axis=0)
# =============================================================================
# =============================================================================
# print(data.dtypes)
# =============================================================================

# =============================================================================
# data = data.fillna('missing')
# =============================================================================
# =============================================================================
# data = data.fillna(0.00001)
# =============================================================================

data['cust_sex'] = data['cust_sex'].fillna('missing')
data['years_old'] = data['years_old'].fillna('missing')

# =============================================================================
# data['is_birth'] = data['is_birth'].fillna('missing')
# data['is_marry'] = data['is_marry'].fillna('missing')
# data['is_job'] = data['is_job'].fillna('missing')
# =============================================================================


data['area_price'] = data['area_price'].fillna(data['area_price'].mean())
data['distance'] = data['distance'].fillna(data['distance'].mean())
data['daohang_dura'] = data['daohang_dura'].fillna(data['daohang_dura'].mean())
data['gongxiangdanche_dura'] = data['gongxiangdanche_dura'].fillna(data['gongxiangdanche_dura'].mean())
data['autohome_dura_201712'] = data['autohome_dura_201712'].fillna(data['autohome_dura_201712'].mean())
data['foot_distance'] = data['foot_distance'].fillna(data['foot_distance'].mean())
data['bike_distance'] = data['bike_distance'].fillna(data['bike_distance'].mean())
data['car_distance'] = data['car_distance'].fillna(data['car_distance'].mean())
data['vis_dealer_cnt'] = data['vis_dealer_cnt'].fillna(0)
data = data.dropna(axis=0)


data['label'] = data['vis_dealer_cnt']
data = data.drop(["vis_dealer_cnt"],axis=1)




# =============================================================================
# data.describe()
# =============================================================================


data = data[data.distance < 20000]
data = data[data.daohang_dura < 200000]
data = data[data.gongxiangdanche_dura < 200000]
data = data[data.autohome_dura_201712 < 200000]


# =============================================================================
# data.describe()
# =============================================================================

target = data['label']


# =============================================================================
# null_counts = data.isnull().sum()
# print(null_counts)
# =============================================================================
# =============================================================================
# print(target.value_counts())
# =============================================================================


from patsy import dmatrices,dmatrix    #凡是分类变量均要使用哑变量处理
CX = dmatrix ('C(years_old) + C(cust_sex) + C(is_birth) + C(is_marry) + C(is_job)',data = data ,return_type = 'dataframe')
XXX=data.drop(["years_old","cust_sex","label","is_birth","is_marry","is_job"], axis=1)
features = CX.join(XXX)


# =============================================================================
# features.describe()
# =============================================================================




from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_new = min_max_scaler.fit_transform(features)

features = pd.DataFrame(features_new, columns=features.columns)




from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
# Lasso
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error



X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.25,random_state=42)


alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

# print(lasso)
print (r2_score_lasso)


mse = mean_squared_error(y_pred_lasso, y_test)
print(mse)
rmse = mse ** (0.5)
print (rmse)



# =============================================================================
# print(lasso.coef_)
# =============================================================================

importances = lasso.coef_
# =============================================================================
# std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
# =============================================================================
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(features.shape[1]):
    print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , features.columns[f] ))














