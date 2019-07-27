# coding=utf-8
##朴素贝叶斯算法：计算先验概率P(c_i)→条件概率P(x_j|c_i)→后验概率##

##原理代码不手写了，主要是把每种算法的原理过一遍，再作一些案例##
"""
# 1、准备数据：从文本中构建词向量
def loadDataSet():
    '''
    创建样本数据
    :return:postList是进行词条切分后的样本集合，classVec是类别标签的集合
    '''
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0,1,0,1,0,1]    # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec
def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])
    #
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return  list(vocabSet)
"""
##基于朴素贝叶斯的垃圾邮件分类##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from nltk.corpus import stopwords    # 导入英文的停用词
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1、读取数据
df = pd.read_csv('data_spam/spam.csv')
# 2、数据处理
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
print(df['label'].value_counts())
## 把label里的标签转换为0、1, ham为正常邮件，spam为垃圾邮件
df['numlabel'] = df['label'].map({'ham': 0, 'spam': 1})
## 统计每一个文本的长度信息
text_length = [len(df.loc[i, 'text']) for i in range(len(df))]
print('the minimum length is ' + str(min(text_length)))
plt.hist(text_length, bins=100, facecolor='blue', alpha=0.5)
# plt.xlim([0,200])    # 设置显示x轴的范围
plt.grid()
plt.show()
## 切分文本，构建文本的向量（基于词频的表示）
stopset = set(stopwords.words('English'))
vectorizer = CountVectorizer(stop_words=stopset, binary=True)
# vectorizer = CountVectorizer()    # 不删除停用词，直接进行切分
x = vectorizer.fit_transform(df.text)
y = df.numlabel

# 3、建立模型
## 切分数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print('训练样本的个数为：' + str(x_train.shape[0]) + '\n' +
      '测试样本的个数为：' + str(x_test.shape[0]))
## 训练模型
clf = MultinomialNB(alpha=1.0,    # Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
                    fit_prior=True)    # fit_prior:Whether to learn class prior probabilities or not.If false, a uniform prior will be used.
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('accuracy on the test data' + str(accuracy_score(y_test, y_pred)))
## 打印混淆矩阵
confusion_matrix(y_test, y_pred, labels=[0, 1])


'''
本文亮点：
①用.map({'a':'b'})将某一列的特征标签a转换为b
文中为：df['numlabel'] = df['label'].map({'ham':0, 'spam':1})
②统计每一行的某个特征下文本的长度，使用迭代器更加简洁
文中为：text_length = [len(df.loc[i,'text']) for i in range(len(df))]
③画直方图plt.hist()
文中为：
import matplotlib.pyplot as plt
plt.hist(text_length, bins=100, facecolor='blue', alpha=0.5)    # bins为每一柱子的宽度，越小越宽
plt.xlim([0,200])    # 设置横坐标的显示范围
plt.axvline(200, hold=None, label='x轴分隔线', c='r', linestyle='--', alpha=0.3)
plt.legend(loc='upper left')
plt.show()
④将英文文本分割为词向量：CountVectorizer()\CountVectorizer(stopwords=stopset,binary=True)
文中为：
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords    # 导入英文的停用词
stopset = set(stopwords.words('English'))
# vectorizer = CountVectorizer()
vectorizer = CountVectorizer(stop_words=stopset, binary=True)
x = vectorizer.fit_transform(df.text)
⑤朴素贝叶斯模型的流程:
文中为：
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
clf = MultinomialNB(alpha=1.0,    # Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
                    fit_prior=True)    # fit_prior:Whether to learn class prior probabilities or not.If false, a uniform prior will be used.
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('accuracy on test data ' + str(accuracy_score(y_test, y_pred)))
'''
