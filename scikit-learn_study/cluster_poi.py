# -*- coding: utf-8 -*-
"""
cluster_poi information
"""



import pandas as pd
import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import json



auto = pd.read_csv('C:/Users/1707500/Desktop/cluster.csv',names = ['clusters_id','clusters_latitude','clusters_longitude','vin'],encoding = 'gb2312')

clusters_latitude = auto['clusters_latitude']
clusters_longitude = auto['clusters_longitude']
status =[]
formatted_address =[]
country =[]
province =[]
city =[]
district =[]
township =[]
poi_name =[]
poi =[]
num = 0




def get_one_page(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
    try:
        response = requests.get(url,headers = headers)                  
        if response.status_code == 200:                     #判断是否爬取网页成功
            return response.text
        return None
    except RequestException:
        return None
    

for i,j in zip(clusters_latitude,clusters_longitude):
    print(num)
    url = 'http://restapi.amap.com/v3/geocode/regeo?key=c8d35a9988ba6acbd7b9ec0209d3c455&location='+str(j)+','+str(i)+'&poitype=&radius=&extensions=base&batch=false&roadlevel='
    html = get_one_page(url)
    data = json.loads(html)

    status.append(data['status'])
    formatted_address.append(data['regeocode']['formatted_address'])
    country.append(data['regeocode']['addressComponent']['country'])
    province.append(data['regeocode']['addressComponent']['province'])
    city.append(data['regeocode']['addressComponent']['city'])
    district.append(data['regeocode']['addressComponent']['district'])
    township.append(data['regeocode']['addressComponent']['township'])
    poi_name.append(data['regeocode']['addressComponent']['neighborhood']['name'])
    poi.append(data['regeocode']['addressComponent']['neighborhood']['type'])


    print(data['status'])
    print(data['regeocode']['formatted_address'])
    print(data['regeocode']['addressComponent']['country'])
    print(data['regeocode']['addressComponent']['province'])
    print(data['regeocode']['addressComponent']['city'])
    print(data['regeocode']['addressComponent']['citycode'])
    print(data['regeocode']['addressComponent']['district'])
    print(data['regeocode']['addressComponent']['adcode'])
    print(data['regeocode']['addressComponent']['township'])
    print(data['regeocode']['addressComponent']['towncode'])
    print(data['regeocode']['addressComponent']['neighborhood']['name'])
    print(data['regeocode']['addressComponent']['neighborhood']['type']) 
    num = num + 1
    
    
    
    
    
print(len(auto['vin']))
print(len(auto['clusters_id']))
print(len(auto['clusters_longitude']))
print(len(auto['clusters_latitude']))
print(len(status))
print(len(formatted_address))
print(len(country))
print(len(province))
print(len(city))
print(len(district))
print(len(township))
print(len(poi_name))
print(len(poi))
df = pd.DataFrame({ 
            'vin' : auto['vin'],                  
            'clusters_id' : auto['clusters_id'],
            'clusters_longitude' :auto['clusters_longitude'],
            'clusters_latitude' : auto['clusters_latitude'],
            'status' : status,
            'formatted_address' : formatted_address,
            'country' : country,
            'province' : province,
            'city' : city,
            'district' : district,
            'township' : township,
            'poi_name' : poi_name,
            'poi' : poi
            })

    
df.to_csv('D:/document/crawling/auto_clusters_poi.csv',index = False)  