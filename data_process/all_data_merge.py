import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import re
weobo=pd.read_excel('../weibo.xlsx')
merge=pd.read_excel('merge.xlsx')
text_data=pd.read_excel('../text_process/topic_complexity.xlsx')
clip_info=pd.read_excel('../clip/results.xlsx')
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


weobo['witchnum']=weobo['witchnum'].apply(witch_num)
weobo['timelength']=weobo['timelength'].apply(time_length)
result=pd.concat([text_data,clip_info],axis=1)
temp=weobo.merge(merge,on='img_name',how="left")
result=result.merge(temp,on='img_name',how="right")

#某列数据为nan使用该列均值替换
def replace_nan_with_mean(column_name,result):
    result[column_name] = result[column_name].fillna(result[column_name].mean())
    return result

#某列数据为0使用该列均值替换
def replace_zero_with_mean(column_name,result):
    result[column_name] = result[column_name].replace(0, result[column_name].mean())
    return result
for j in ['EC','OC', 'colorfulness', 'brightness', 'quality','contrast','similarity',
        'timelength', 'topic_complexity']:
    result=replace_nan_with_mean(j,result)
    result=replace_zero_with_mean(j,result)


# 如果需要，可以重置索引
result = result.reset_index(drop=True)
# result['witchnum']=group_witcgnum(result['witchnum'])
result.to_excel('../all_data.xlsx', index=False)