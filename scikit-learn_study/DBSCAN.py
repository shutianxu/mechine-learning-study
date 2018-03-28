# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 08:43:29 2018

@author: 1707500
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler



# #############################################################################
# Generate sample data
auto = pd.read_csv('C:/Users/1707500/Desktop/auto.csv',encoding = 'gb2312')

auto = auto[['vin','ignition_time','stop_longitude','stop_latitude']]

vin_list = list(set(auto['vin']))
clusters= []
data1 =  pd.DataFrame()
# =============================================================================
# data2 =  pd.DataFrame()
# =============================================================================
ll = []
ls =[]
clusters_longitude = []
clusters_latitude = []


def avg(x):
    if len(x) < 1:
        return x[0]
    else:
        return sum(x) / len(x)   

for i in vin_list[:20]:
    X = auto[auto.vin == i]
    Y = X[['vin','ignition_time']]
    X = X[['stop_longitude','stop_latitude']]    
    try:
        
        db = DBSCAN(eps=0.001, min_samples=5)
        db.fit(X)
        labels = db.labels_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique) - (1 if -1 in labels else 0)
        print("number of estimated clusters : %d" % n_clusters_)
    except:
        print("number of estimated clusters : 0")
        
        

# =============================================================================
# #噪音点评估
# raito = len(labels[labels[:] == -1]) / len(labels)
# print('Noise raito:', format(raito, '.2%'))
# =============================================================================
    '''
    聚类画图调参
    '''
    
    try:
        if len(X['stop_longitude']) == len(labels):
            X['label'] = labels
            
            
        
        plt.figure(1)
        plt.clf()
        plt.figure(figsize=(12,12)) 
        colors = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF', '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF', '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700', '#DAA520', '#808080', '#008000', '#ADFF2F', '#F0FFF0', '#FF69B4', '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370DB', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970', '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000', '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE', '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD', '#B0E0E6', '#800080', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#FAA460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090', '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3', '#FFFFFF', '#F5F5F5', '#FFFF00', '#9ACD32']
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            

            plt.plot(X[X.label == k]['stop_longitude'], X[X.label == k]['stop_latitude'],'o', markerfacecolor = col, markersize=10)
            plt.plot(avg(X[X.label == k]['stop_longitude']), avg(X[X.label == k]['stop_latitude']),'*', markerfacecolor = 'b', markersize=10)
    # =============================================================================
    #         plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor = 'y', markersize=20)
    # =============================================================================
      
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
        for j in range(n_clusters_):
            ll.append(i)
            ls.append(j)
            clusters_longitude.append(avg(X[X.label == j]['stop_longitude']))
            clusters_latitude.append(avg(X[X.label == j]['stop_latitude']))  
            
            
            
# =============================================================================
#             #仅测试使用
#             print(ll)
#             print(ls)
#             print(clusters_longitude)
#             print(clusters_latitude)
# 
# =============================================================================
        
        
        
        data2 = pd.DataFrame({ 
                    'vin' : ll,                  
                    'clusters_id' : ls,
                    'clusters_longitude' : clusters_longitude,
                    'clusters_latitude' : clusters_latitude          
                    })
    except:
        pass


    '''
    文件输出 output1
    '''    
    df1 = pd.DataFrame({ 'vin' : Y['vin'],
                        'ignition_time' : Y['ignition_time'],
                        'stop_longitude' : X['stop_longitude'],
                        'stop_latitude' : X['stop_latitude'],                     
                        'clusters_id' : labels
                         })
    data1 = pd.concat([data1,df1])
    
    
    
    
    
#文件1输出    
data1.to_csv('D:/document/crawling/auto_clusters1.csv',index = False)  
data2.to_csv('D:/document/crawling/auto_clusters2.csv',index = False)  
    
    
    

# =============================================================================
# data = data1[data1.clusters_id != -1]
# 
# for i in vin_list[:20]:
#     X = auto[auto.vin == i]
#     Y = X[['vin']]
#     X = X[['stop_longitude','stop_latitude']]
#     try:
#         bandwidth = estimate_bandwidth(X, quantile=0.2)
#         ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#         ms.fit(X)
#         
#         labels_unique_new = np.unique(ms.labels_)
#         n_clusters_new = len(labels_unique_new)
#         print("number of estimated clusters : %d" % n_clusters_new)
#     except:
#         print("number of estimated clusters : 0")
#         
#     labels = ms.labels_
#     for j in range(len(set(ms.labels_))):
#         ll.append(j)
#         ls.append(i)
#         
#         
#     for k in ms.cluster_centers_:
#         clusters_longitude.append(k[0])
#         clusters_latitude.append(k[1])
# 
#     '''
#     文件输出
#     '''     
#     df2 = pd.DataFrame({ 'vin' : ls,
#                         'clusters_id' : ll,
#                         'clusters_longitude' : clusters_longitude,
#                         'clusters_latitude' : clusters_latitude
#                          })
#     
#     data2 = pd.concat([data2,df2])
# data2.to_csv('D:/document/crawling/auto_clusters2.csv')        
# =============================================================================

  

