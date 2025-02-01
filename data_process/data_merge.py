import json
import  pandas as pd
import math
import re
with open('projects_glm.json', 'r', encoding='utf-8') as file:
    data_glm = json.load(file)
with open('projects_yolo.json', 'r', encoding='utf-8') as file:
    data_yolo = json.load(file)
data_calculate=pd.read_excel('Picture_features.xlsx').values.tolist()
yolo_EC_OC=[]
for j in data_yolo:
    cup_calss = ['cup', 'bottle']
    data_info = json.loads(j)
    image_name = list(data_info.keys())[0]
    info_list = data_info[image_name]
    cup,person,EC_CUP_PERSON,OC=0,0,0,0
    if len(info_list)>0:
        for item in info_list:
            for key in item.keys():
                OC += float(item[key]) * math.log(float(item[key])) * -1
                if key in cup_calss:
                    cup = 1
                if key == 'person':
                    person = 1
    else:
        OC=0
    EC_CUP_PERSON = cup + person
    yolo_EC_OC.append([image_name,EC_CUP_PERSON,OC])
glm_list=[list(x.items())[0] for x in data_glm]

merge=[[x[0],y[1]+eval(z[1]),y[2],x[1],x[2],x[3],x[4]] for x in data_calculate for y in yolo_EC_OC for z in glm_list if x[0]==y[0]==z[0]]
pd.DataFrame(merge,columns=['image_name','EC','OC','colorfulness', 'brightness', 'quality','contrast']).to_excel('merge.xlsx',index=False)


