# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
v1.0购车重要因素分析
v1.1增加部分数据处理，用众数，平均值，0取代NAN值
v1.2调整部分参数
V2.0增加用户画像建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = ['device_number' ,'cust_sex' ,'years_old' ,'shizhengyaowen' ,'tiyuzixun' ,'kejizixun' ,'yulezixun' ,'caijingzixun' ,'wenhuajiaoyu' ,'qiyemenhu' ,'shehuiziyuan' ,'zonghezixun' ,'xinwenzixun' ,'jishitongxun' ,'hunlianjiaoyou' ,'shequluntan' ,'shejiaowangluo' ,'tongxinjiaoliu' ,'shoujishipin' ,'shoujiyuedu' ,'shoujiyinpin' ,'shoujiyouxi' ,'yulegaoxiao' ,'sheyingtupian' ,'wangluodiantai' ,'yingyuanpiaowu' ,'shijiantianqi' ,'sousuoyinqin' ,'dahangwangzhan' ,'shenghuozonghe' ,'jiajufuwu' ,'qichexinxi' ,'muyingzixun' ,'yiliaojiankang' ,'lvyouchuxing' ,'wuliukuaidi' ,'canyinmeishi' ,'feifawangzhan' ,'waimaisongcan' ,'ditudaohang' ,'dacheruanjian' ,'chaxunfuwu' ,'tuangoudazhe' ,'jiaoyuxuexi' ,'wangshanggouwu' ,'dianzizhifu' ,'shangwubangong' ,'qiuzhizhaopin' ,'yingxiaopingtai' ,'yunfuwu' ,'shoujiyinhang' ,'jinronglicai' ,'wangshangyingyeting' ,'anquanshadu' ,'liulanqi' ,'xitonggongju' ,'shurufa' ,'cidianfanyi' ,'zhutizhuomian' ,'yingyongshangdian' ,'wangluocunchu' ,'wuxianguanli' ,'dianziyoujian' ,'gongjuruanjian','car_owner' ,'area_price' ,'distance' ,'daohang_cnt' ,'daohang_dura' ,'didi_cnt' ,'didi_dura' ,'autohome_cnt' ,'autohome_dura' ,'gongxiangdanche_cnt' ,'gongxiangdanche_dura' ,'autohome_cnt_201712' ,'autohome_dura_201712' ,'autohome_cnt_201711' ,'autohome_dura_201711' ,'gongxiangqiche_cnt' ,'gongxiangqiche_dura' ,'foot_distance' ,'bike_distance' ,'car_distance' ,'vis_dealer_cnt' ,'cnxiandai_dealer_cnt' ,'gas_vis_cnt']
data = pd.read_csv("C:/Users/1707500/Desktop/cusc_data_new.csv", sep = '\t',encoding = 'gb2312',names = header)

# =============================================================================
# print(df)
# =============================================================================
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
data['car_owner'] = data['car_owner'].fillna(0)

# =============================================================================
# data['area_price'] = data['area_price'].fillna(data['area_price'].mean())
# data['distance'] = data['distance'].fillna(data['distance'].mean())
# =============================================================================

# =============================================================================
# print(data['area_price'].value_counts())
# print(data['distance'].value_counts())
# =============================================================================


data['area_price'] = data['area_price'].fillna(data['area_price'].median())
data['distance'] = data['distance'].fillna(data['area_price'].median())
data['daohang_dura'] = data['daohang_dura'].fillna(0)
data['gongxiangdanche_dura'] = data['gongxiangdanche_dura'].fillna(0)
data['autohome_dura_201712'] = data['autohome_dura_201712'].fillna(0)
data['foot_distance'] = data['foot_distance'].fillna(data['foot_distance'].median())
data['car_distance'] = data['car_distance'].fillna(data['car_distance'].median())
data['bike_distance'] = data['bike_distance'].fillna(data['bike_distance'].median())




data['vis_dealer_cnt'] = data['vis_dealer_cnt'].fillna(0)
data = data.dropna(axis=0)


data['label'] = data['vis_dealer_cnt'].map(lambda x: 1 if x >= 1 else 0)
data = data.drop(["vis_dealer_cnt"],axis=1)

# =============================================================================
# null_counts = data.isnull().sum()
# print(null_counts)
# =============================================================================

# =============================================================================
# data.describe()
# =============================================================================

data = data[data.car_owner != 1]
# =============================================================================
# data = data[data.distance < 50000]
# data = data[data.daohang_dura < 1000000]
# data = data[data.gongxiangdanche_dura < 1000000]
# data = data[data.autohome_dura_201712 < 1000000]
# =============================================================================

orig_columns = data.columns
drop_columns = []
for col in orig_columns:
    col_series = data[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
data = data.drop(drop_columns, axis=1)
print(drop_columns)

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
CX = dmatrix ('C(years_old) + C(cust_sex)',data = data ,return_type = 'dataframe')
XXX=data.drop(["years_old","cust_sex","label"], axis=1)
features = CX.join(XXX)

features = features.drop(['C(cust_sex)[T.missing]','C(years_old)[T.missing]'],axis=1)
# =============================================================================
# features.describe()
# =============================================================================




from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_new = min_max_scaler.fit_transform(features)

features = pd.DataFrame(features_new, columns=features.columns)




from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report   
from sklearn import metrics



X_train,X_test,y_train,y_test = train_test_split(
features,target,test_size=0.2,random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=50 ,min_samples_split=1000, random_state=42,class_weight="balanced")
clf = clf.fit(X_train, y_train) 
y_pre = clf.predict(X_test)

print(classification_report(y_test,y_pre))
print(metrics.roc_auc_score(y_test,y_pre)) #预测Y值得分
        
num = 1        
for j in [50,100]:
    for i in range(100):
        if ((i%5 == 0) and i != 0):
            for k in range(3001):
                if ((k%1000 ==0) and (k != 0)):
                    print("Index: %d" %num)
                    print("n_estimators:%d" % j)
                    print("max_depth:%d" % (i))
                    print("min_samples_split:%d" % (k))
                    clf = RandomForestClassifier(n_estimators=j, max_depth=i ,min_samples_split=k, random_state=42,class_weight="balanced")
                    clf = clf.fit(X_train, y_train) 
                    y_pre = clf.predict(X_test)

                    print(classification_report(y_test,y_pre))
                    print(metrics.roc_auc_score(y_test,y_pre)) #预测Y值得分
                    
                    importances = clf.feature_importances_
                    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
                    indices = np.argsort(importances)[::-1]
                    # Print the feature ranking
                    print("Feature ranking:")
                    for f in range(features.shape[1]):
                        print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , features.columns[indices[f]] ))

                    with open('D:/document/crawling/random_forest_vis_dealer_cust_internet.txt','a') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！                        
                        f.write('\n')
                        f.write('\n')
                        f.write("Index: %d" %num)
                        f.write('\n')
                        f.write("n_estimators:%d" % j)
                        f.write('\n')
                        f.write("max_depth:%d" % (i))
                        f.write('\n')
                        f.write("min_samples_split:%d" % (k))
                        f.write('\n')
                        f.write(classification_report(y_test,y_pre))                    
                        f.write(str(metrics.roc_auc_score(y_test,y_pre)))
                        f.write('\n')
                        f.write('\n')
                        f.write('Feature ranking:')
                        for ll in range(features.shape[1]):
                            f.write(str("%d. feature %d (%f): %s" % (ll + 1, indices[ll], importances[indices[ll]] , features.columns[indices[ll]] )))
                            f.write('\n')
                        f.write('\n')
                    num = num + 1






# =============================================================================
# importances = clf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
# indices = np.argsort(importances)[::-1]
# print("Feature ranking:")
# for f in range(features.shape[1]):
#     print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , features.columns[indices[f]] ))
# =============================================================================




