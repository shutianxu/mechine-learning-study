# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
v1.0汽车之家时间序列
v2.0汽车之家时间序列、用户特征建模分析
v3.0加载汽车之家时间序列模型、加载用户特征模型
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.externals import joblib

# =============================================================================
# header = ['device_num',	'-8'	,'-7'	,'-6'	,'-5'	,'-4'	,'-3'	,'-2'	,'-1']
# header_1 = ['device_num','vis_dealer_y']
# data_autohome = pd.read_csv("C:/Users/GZ-User01/Desktop/deler_vis_max_sec.csv", sep = '\t',names = header)
# vis_dealer = pd.read_csv("C:/Users/GZ-User01/Desktop/deler_vis_y.csv", sep = '\t',names = header_1)
# data=pd.merge(data_autohome,vis_dealer,on='device_num',how='left')
# =============================================================================

header_2 = ['device_num','vis_dealer_y']
data_label = pd.read_csv('C:/Users/GZ-User01/Desktop/deler_vis_y.csv', sep = '\t',names = header_2)


header = ['device_num',	'-1'	,'-2'	,'-3'	,'-4']
data_autohome = pd.read_csv('C:/Users/GZ-User01/Desktop/autohome_week_4_max_sec.csv', sep = '\t',names = header)



header_1 = ['device_num' ,'cust_sex' ,'years_old' ,'yingyongshangdian_dura' ,'tuangoudazhe_dura' ,'shangwubangong_dura' ,'jishitongxun_dura' ,'dacheruanjian_dura' ,'car_owner' ,'area_price' ,'distance' ,'daohang_cnt' ,'daohang_dura' ,'didi_cnt' ,'didi_dura' ,'autohome_cnt' ,'autohome_dura' ,'gongxiangdanche_cnt' ,'gongxiangdanche_dura' ,'autohome_cnt_201712' ,'autohome_dura_201712' ,'autohome_cnt_201711' ,'autohome_dura_201711' ,'gongxiangqiche_cnt' ,'gongxiangqiche_dura' ,'yinyue_cnt' ,'yinyue_dura' ,'zhifubao_cnt' ,'zhifubao_dura' ,'wechat_cnt' ,'wechat_dura' ,'taobao_cnt' ,'taobao_dura' ,'zhihu_cnt' ,'zhihu_dura' ,'meituan_cnt' ,'meituan_dura' ,'mayi_cnt' ,'mayi_dura' ,'foot_distance' ,'bike_distance' ,'car_distance' ,'vis_deal_cnt' ,'xiandai_dealer_cnt' ,'gas_vis_cnt']
data_cust = pd.read_csv('C:/Users/GZ-User01/Desktop/cusc_data_02.csv', sep = '\t',names = header_1)

data_1=pd.merge(data_cust,data_autohome,on='device_num',how='left')
data = pd.merge(data_1,data_label,on='device_num',how='left')
data = data.drop(["didi_cnt","autohome_cnt_201711","gongxiangqiche_cnt","gas_vis_cnt","xiandai_dealer_cnt","daohang_cnt","autohome_cnt"],axis=1)
data = data.drop(["gongxiangdanche_cnt","yinyue_cnt","zhifubao_dura","wechat_cnt","autohome_cnt_201712","taobao_cnt","zhihu_cnt","meituan_dura","mayi_cnt"],axis=1)
data = data[data.autohome_dura_201712 > 0]
data = data.drop(["meituan_cnt","vis_deal_cnt","autohome_dura","autohome_dura_201712","autohome_dura_201711"],axis=1)

data['vis_dealer_y'] = data['vis_dealer_y'].map({u'Y':1})
data=data.fillna(0)
data['life_stage'] = 0
data.to_csv("C:/Users/GZ-User01/Desktop/useless_table.csv",index = False)

target = data['vis_dealer_y']
features = pd.get_dummies(data, columns = ['cust_sex','years_old'])
features = features.drop(['vis_dealer_y'],axis=1)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report   
from sklearn import metrics



X_train_1,X_test_1,y_train,y_test = train_test_split(
features,target,test_size=0.25,random_state=2018)

X_train = X_train_1.drop(['device_num'],axis=1)
X_test = X_test_1.drop(['device_num'],axis=1)


'''
lightgbm
'''
print("LGB test")
clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=150, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=2,
    learning_rate=0.3, min_child_weight=50, random_state=2018, n_jobs=-1,class_weight = 'balanced'
)

clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='auc',early_stopping_rounds=100)

y_pre = clf.predict(X_test)
y_pre_pro = clf.predict_proba(X_test)[:, 1]


df = pd.DataFrame({
        'device_num':X_test_1['device_num'],
        'y_pre':y_pre,
        'y_pre_pro':y_pre_pro
                })

df.to_csv('C:/Users/GZ-User01/Desktop/final_1.csv',index = False)

# =============================================================================
# print(classification_report(y_test,y_pre))
# print(metrics.roc_auc_score(y_test,y_pre))  #预测Y值得分
# =============================================================================
 
joblib.dump(clf,"C:/Users/GZ-User01/Desktop/model.m")







