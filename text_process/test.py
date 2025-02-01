# import re
# def count_non_chinese_english_digits_punctuation(text):
#     # 定义正则表达式，匹配中文、英文、数字和常见标点符号
#     # \u4e00-\u9fa5 匹配中文汉字
#     # a-zA-Z 匹配英文字母
#     # 0-9 匹配数字
#     # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 匹配常见英文标点符号
#     # ，。、；：？！“”‘’《》【】〔〕（）—…·﹏﹏─ 匹配常见中文标点符号
#     chinese_par=u'[\u4e00-\u9fa5]'
#     english_par=u'[a-zA-Z\w\s]'
#     digit_par=u'[0-9]'
#     punctuation_par=u'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。、；：？！“”‘’《》【】〔〕（）—…·﹏﹏─]'
#     # 使用 re.sub 函数将匹配的字符替换为空字符串
#     text = re.sub(chinese_par, '', text)
#     text = re.sub(english_par, '', text)
#     text = re.sub(digit_par, '', text)
#     text = re.sub(punctuation_par, '', text)
#     print(text)
#     # 统计剩余字符的数量
#     return len(text)
#
# text='#茶颜悦色[超话]# 🐍茶颜蛇年上新🆕速来接好运❗️ 茶颜给大家拜早年啦🧨 巳蛇🐍出来接班辰🐲 也给大家带来了蛇年主题新周边🎁 【今年有福兮】字字皆祝福~ 新年一起交好运吧~~~ #茶颜悦色##茶颜家族##茶颜新年礼##茶颜陀螺大王##长沙吃喝玩乐##蛇年#'
# print(count_non_chinese_english_digits_punctuation(text))

import numpy as np
from sklearn.cluster import KMeans


# 生成一些示例一维数据
data = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]).reshape(-1, 1)

# 设置簇的数量
k = 3

# 使用K - means算法进行聚类
kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

# 获取每个数据点所属的簇标签
labels = kmeans.labels_

# 获取簇中心
cluster_centers = kmeans.cluster_centers_

# 将每个数据点替换为其所属簇的中心，实现离散化
discretized_data = np.array([cluster_centers[label] for label in labels])

print("原始数据:", data.flatten())
print("离散化后的数据:", discretized_data.flatten())
