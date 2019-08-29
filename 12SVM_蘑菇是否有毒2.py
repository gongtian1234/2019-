# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# 使用的数据集: https://archive.ics.uci.edu/ml/datasets/mushroom
df = pd.read_csv('mushrooms.csv')
print(df.shape)
print(df.isnull().sum())    # 没有缺失值

# class列进行转换：p：有毒；e: 可以吃
df['class'] = df['class'].map({'p': 0, 'e': 1})

## 第一种方法：不进行特征选择，直接将数据one-hot编码，然后建立模型 ##
df1 = df.copy()
df1 = pd.get_dummies(df1.loc[:, df1.columns!='class'])
df1['class'] = df['class']
# 建立模型
x = df1.loc[:, df1.columns!='class']
y = df1['class']
x_train, x_test, y_train, y_test = train_test_split(x, y)
svm = SVC(class_weight='balanced')
params = {'C': [1,5,15,20,25,30,50],
          'kernel': ['linear', 'poly', 'rbf']}
grid_search = GridSearchCV(svm, param_grid=params, n_jobs=2, cv=10)
grid_search = grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
model = grid_search.best_estimator_
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))    # 这个结果很奇怪，准确率等各个指标竟然都是1

## 对特征进行筛选，用堆叠图初略进行特征筛选 ##
# 由于所有的特征都是类别型，所以依次对其绘制堆叠图
df2 = df.copy()
def stacked_plot(x, y, xlabel):
    table = pd.crosstab(x, y)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.xlabel(xlabel)
    plt.ylabel('y')
    plt.show()
columns = list(df2.columns)
columns.remove('class')
for clst in columns:
    stacked_plot(df2[clst], df2['class'], xlabel=clst)
## 发现veil-type只有一类p，所以将其舍弃
df2.drop('veil-type', axis=1, inplace=True)
# 将类别型数据转为one-hot编码
df2 = pd.get_dummies(df2.loc[:, df2.columns!='class'])
df2['class'] = df['class']
# 训练模型
x = df2.loc[:, df2.columns!='class']
y = df2['class']
x_train, x_test, y_train, y_test = train_test_split(x, y)
svm = SVC(class_weight='balanced')
params = {'C': [1, 5, 10, 15, 20, 30, 50],
          'kernel': ['linear', 'poly', 'rbf']}
grid_search = GridSearchCV(svm, param_grid=params, n_jobs=2, cv=10)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
model = grid_search.best_estimator_
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

## 如果用逻辑回归做，准确率还是1吗？？
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

'''
本文亮点：
①选择除了y以外的全部列：df.loc[:, df.columns!='y']
文中为：df1=pd.get_dummies(df1.loc[:, df1.columns!='class'])
②得到精确率和召回率的报告：
from sklearn.metrics import classification_report
classification_report(y_test, y_pred)
'''

