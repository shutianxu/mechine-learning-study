import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle

# #############################################################################
# Generate sample data
auto = pd.read_csv('C:/Users/1707500/Desktop/auto.csv',encoding = 'gb2312')

auto = auto[['vin','stop_longitude','stop_latitude']]

vin_list = list(set(auto['vin']))
clusters= []

for i in vin_list[:20]:
    X = auto[auto.vin == i]
    X = X[['stop_longitude','stop_latitude']]    
    try:
        bandwidth = estimate_bandwidth(X, quantile=0.2)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        
        cluster_centers = ms.cluster_centers_
        y_pred = ms.predict(X)
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("number of estimated clusters : %d" % n_clusters_)
    except:
        print("number of estimated clusters : 0")
# =============================================================================
#     clusters.append(n_clusters_)
# =============================================================================
    try:
        if len(X['stop_longitude']) == len(y_pred):
            X['label'] = y_pred
            
            
        
        plt.figure(1)
        plt.clf()
        plt.figure(figsize=(12,12)) 
    # =============================================================================
    #     colors = cycle('bgrcmk')
    # =============================================================================
        colors = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF', '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF', '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700', '#DAA520', '#808080', '#008000', '#ADFF2F', '#F0FFF0', '#FF69B4', '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370DB', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970', '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000', '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE', '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD', '#B0E0E6', '#800080', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#FAA460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090', '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3', '#FFFFFF', '#F5F5F5', '#FFFF00', '#9ACD32']
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(X[X.label == k]['stop_longitude'], X[X.label == k]['stop_latitude'],'o', markerfacecolor = col, markersize=10)
            
    # =============================================================================
    #         plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor = 'y', markersize=20)
    # =============================================================================
      
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    except:
        pass








# =============================================================================
# df = pd.DataFrame({ 'vin' : vin_list,
#                     'clusters' : clusters
#                          })
# 
#     
# df.to_csv('D:/document/crawling/auto_clusters.csv')    
# =============================================================================
