import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import re
weobo=pd.read_excel('../weibo.xlsx')
merge=pd.read_excel('merge.xlsx')
text_data=pd.read_excel('../text_process/topic_complexity.xlsx')
def witch_num(text):
    if type(text)==float:
        return 0
    else:
        text = re.sub(r'次观看', '', text)
        if text[-1] == '万':
            return eval(text[:-1]) * 10000
        else:
            return eval(text)
def time_length(text):
    if type(text)==float:
        return 0
    else:
        h_m=text.split(':')
        return int(h_m[0])*60+int(h_m[1])
def k_mean_process(data):
    k=5
    # 使用K - means算法进行聚类
    kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(data).reshape(-1, 1))

    # 获取每个数据点所属的簇标签
    labels = kmeans.labels_

    # 获取簇中心
    cluster_centers = kmeans.cluster_centers_
    discretized_data = np.array([cluster_centers[label] for label in labels])
    return discretized_data


weobo['witchnum']=weobo['witchnum'].apply(witch_num)
weobo['timelength']=weobo['timelength'].apply(time_length)

result = pd.concat([weobo, merge, text_data], axis=1)
# 假设你想删除 'column_name' 列中值为0的整行数据
column_name = 'witchnum'  # 替换为你的列名

# 找到值为0的行
rows_to_drop = result[result[column_name] == 0].index

# 删除这些行
result = result.drop(rows_to_drop)

# 如果需要，可以重置索引
result = result.reset_index(drop=True)
# result['witchnum']=k_mean_process(result['witchnum'])

# result[column_name] = pd.qcut(result[column_name] , q=10, labels=[x for x in range(1, 11)])
result.to_excel('../all_data.xlsx', index=False)