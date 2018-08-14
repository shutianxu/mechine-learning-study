# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
v1.0汽车之家时间序列
v2.0汽车之家时间序列、用户特征建模分析
v3.0加载汽车之家时间序列模型、加载用户特征模型,lightgbm模型
v4.0改用逻辑回归
"""

import pandas as pd
from sklearn.externals import joblib
import sys
import os
pwd = os.getcwd()


Prov_Id = sys.argv[1]
Month_Id = sys.argv[2]
Time_Data = sys.argv[3]
Area_Id = sys.argv[4]



header_2 = ['device_num','-1','-2','-3','-4']

data_label = pd.read_csv(open(pwd + "/output/data/info/cust_profile_info_{0}_{1}_{2}_{3}.csv".format(Time_Data,Month_Id,Prov_Id,Area_Id),'rU'), sep = '\t',names = header_2)


header_1 = ['device_num', 'cust_sex', 'years_old', 'car_owner', 'area_price', 'distance', 'daohang_cnt', 'daohang_dura', 'didi_cnt', 'didi_dura', 'gongxiangdanche_cnt', 'gongxiangdanche_dura', 'gongxiangqiche_cnt', 'gongxiangqiche_dura', 'foot_distance', 'bike_distance', 'car_distance', 'gas_vis_cnt', 'lift_stage_yn']


data_cust = pd.read_csv(open(pwd + "/output/data/visiter_yn/dealer_visiter_yn_{0}_{1}_{2}_{3}.csv".format(Time_Data,Month_Id,Prov_Id,Area_Id),'rU')
, sep = '\t',names = header_1,error_bad_lines=False,quoting=3)


header_3 = ['device_num','-1','-2','-3','-4', 'misidn','cust_sex', 'years_old', 'car_owner', 'area_price', 'distance', 'daohang_cnt', 'daohang_dura', 'didi_cnt', 'didi_dura', 'gongxiangdanche_cnt', 'gongxiangdanche_dura', 'gongxiangqiche_cnt', 'gongxiangqiche_dura', 'foot_distance', 'bike_distance', 'car_distance', 'gas_vis_cnt', 'lift_stage_yn']

#当前左链接后的两个表的数据
data_left_join = pd.read_csv(open(pwd + "/output/data/test/left_join_data_{0}_{1}_{2}_{3}.csv".format(Time_Data,Month_Id,Prov_Id,Area_Id),'rU')
, sep = '\t',names = header_3,error_bad_lines=False,quoting=3) 


data_left_join['lift_stage_yn'] = data_left_join['lift_stage_yn'].map({u'Y':1,u'N':0})
data_left_join=data_left_join.fillna(0)

features = data_left_join[['-1', '-2', '-3', '-4','cust_sex', 'years_old','car_owner','area_price','distance','daohang_dura','gongxiangqiche_dura','gongxiangdanche_dura','didi_dura','foot_distance','bike_distance','car_distance','lift_stage_yn']]

features = pd.get_dummies(features, columns = ['cust_sex','years_old'])

print(features.columns)

##基于树的方法不用做标准化、归一化处理
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_new = min_max_scaler.fit_transform(features)



features_1 = features[['-1', '-2', '-3', '-4']]

#print(features_1)
features_2 = features[['car_owner','area_price','distance','daohang_dura',
       'gongxiangqiche_dura', 'gongxiangdanche_dura', 'didi_dura',
       'foot_distance','bike_distance', 'car_distance','lift_stage_yn',
       'cust_sex_0.0','cust_sex_1.0', 'cust_sex_2.0','years_old_0',
       'years_old_0-20old','years_old_20-30old','years_old_30-40old',
       'years_old_40-50old','years_old_50-60old','years_old_60-70old']]



clf_1 = joblib.load(pwd + '/npy/autohome_model.m') 
y_pre_pro_1 = clf_1.predict_proba(features_1)[:, 1]


clf_2 = joblib.load(pwd + '/npy/cust_feature_model.m') 
y_pre_pro_2 = clf_2.predict_proba(features_2)[:, 1]



df = pd.DataFrame({
         'device_num':data['device_num'],
         'autohome_model_y_pre_pro':y_pre_pro_1,
         'user_feature_model_y_pre_pro':y_pre_pro_2
                })


decimals = pd.Series([2, 2], index=['autohome_model_y_pre_pro', 'user_feature_model_y_pre_pro'])
df = df.round(decimals)

df.to_csv(pwd + "/output/output_data/data_{0}_{1}_{2}_{3}.csv".format(Time_Data,Month_Id,Prov_Id,Area_Id))
