import math

import jieba
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
def count_non_chinese_english_digits_punctuation(text):
    chinese_par=u'[\u4e00-\u9fa5]'
    english_par=u'[a-zA-Z\w\s]'
    digit_par=u'[0-9]'
    punctuation_par=u'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。、；：？！“”‘’《》【】〔〕（）—…·﹏﹏─]'
    # 使用 re.sub 函数将匹配的字符替换为空字符串
    text = re.sub(chinese_par, '', text)
    text = re.sub(english_par, '', text)
    text = re.sub(digit_par, '', text)
    text = re.sub(punctuation_par, '', text)
    return len(text)

def filter_non_chinese_english(text):
    # 正则表达式匹配中文和英文
    pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z]+')
    # 查找所有匹配的内容
    result = pattern.findall(text)
    # 将匹配结果拼接成字符串
    filtered_text = ''.join(result)
    return filtered_text
# 1. 数据预处理
def preprocess(texts, stopwords):
    processed_texts = []
    for text in texts:
        # 分词
        words = jieba.lcut(filter_non_chinese_english(text))
        # 去除停用词
        filtered_words = [word for word in words if word not in stopwords]
        processed_texts.append(filtered_words)
    return processed_texts

def coherence(num_topics):
    ldamodel = LdaModel(corpus_bow, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    ldacm = CoherenceModel(model=ldamodel, texts=processed_corpus, dictionary=dictionary, coherence='c_v')
    print(ldacm.get_coherence())
    return ldacm.get_coherence()


def calculate_topic_complexity(title):
    # 预处理标题
    processed_title = preprocess(title,stopwords)
    # 将标题转换为词袋表示
    bow_title = dictionary.doc2bow(processed_title)
    # 获取标题的主题分布
    topic_distribution = lda_model.get_document_topics(bow_title)
    # 计算主题复杂度
    topic_complexity = 0
    for _, prob in topic_distribution:
        if prob > 0:
            topic_complexity -= prob * math.log(prob)
    return topic_complexity

if __name__ == '__main__':
    # 示例中文语料
    corpus = pd.read_excel('../weibo.xlsx')['title'].tolist()
    # 运用列表推导式读取文件内容并添加到列表中
    stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8')]
    # 加载自定义词典
    jieba.load_userdict('save_word.txt')

    processed_corpus = preprocess(corpus, stopwords)
    # print(processed_corpus)
    # 2. 构建词典和语料库
    dictionary = corpora.Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]

    # x = range(20, 61)
    # y = [coherence(i) for i in x]
    # # 设置支持中文的字体
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以根据系统选择合适的字体，如SimHei（黑体）、Microsoft YaHei（微软雅黑）等
    # # 解决负号显示问题
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.plot(x, y)
    # plt.xlabel('主题数目')
    # plt.ylabel('coherence大小')
    # plt.title('主题-coherence变化情况')
    # plt.show()

    lda_model = LdaModel(corpus=corpus_bow,
                         id2word=dictionary,
                         num_topics=47,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
    topicomplexity=[]
    for title in processed_corpus:
        bow_title = dictionary.doc2bow(title)
        topic_distribution = lda_model.get_document_topics(bow_title)
        # 计算主题复杂度
        topic_complexity = 0
        for _, prob in topic_distribution:
            if prob > 0:
                topic_complexity -= prob * math.log(prob)
        emo=count_non_chinese_english_digits_punctuation(corpus[processed_corpus.index(title)])
        topicomplexity.append([title,topic_complexity,emo])
    pd.DataFrame(topicomplexity,columns=['title','topic_complexity','emoji_num']).to_excel('topic_complexity.xlsx',index=False)



