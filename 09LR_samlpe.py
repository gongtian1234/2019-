# coding=utf-8
'''
本案例仅用于了解线性回归模型的训练和使用，对于进行股票预测没有实际意义（股票预测可以考虑用RNN等网络模型）
所以又找了黑马班的线性回归的例子——波士顿房价预测
'''

## 小插曲：股票 ##
### 有下载股票数据和画k线图的代码
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 用以下代码下载一个公司的股票交易历史
import tushare as ts
df = ts.get_hist_data('601001')    # 仅可获取六个月以内的数据
df.to_csv('./data/601001大同煤业.csv')
print('数据的维度为', df.shape)
df.reset_index(inplace=True)
## 让数据按照时间升序排列
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by='date', inplace=True, ascending=True)
## 查看一下是否有缺失数据，有进行缺失数据处理
print('查看缺失数据（未进行缺失数据处理前）', df.isnull().sum())
# 发现没有缺失数据，如果有可用一下代码
# df.dropna(axis=0, inplace=True)
## 画k线图
min_date = df['date'].min()
max_date = df['date'].max()
print('first day is ', min_date)
print('last day is ', max_date)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
init_notebook_mode()
trace = go.Ohlc(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
data = [trace]
py.iplot(data, filename='E:\python\实战\机器学习实战\Cp5_logisticRegresson/simple_ohlc.html', image='png')
'''

# 利用sklearn自由数据集进行波士顿房价预测的案例
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

## 获取数据
df = load_boston()
# print('sklear自有数据集——波士顿房价数据的维度', df.shape)
x_train, x_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.25)
## 数据进行标准化，以消除量纲
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)
std_y = StandardScaler()
y_train = std_y.fit_transform(y_train.reshape(-1,1))
y_test = std_y.transform(y_test.reshape(-1,1))

# 1、建立OLS的线性回归模型
lr = LinearRegression()
lr.fit(x_train, y_train)
print('OLS的线性回归模型的系数', lr.coef_)
y_pred1 = lr.predict(x_test)
print('线性回归的均方误差为：', mean_squared_error(std_y.inverse_transform(y_test), std_y.inverse_transform(y_pred1)))

# 2、建立梯度下降的线性回归模型
SGDr = SGDRegressor()
SGDr.fit(x_train, y_train)
print('SGDr的线性回归系数为：', SGDr.coef_)
y_pred2 = SGDr.predict(x_test)
print('SGDr的均方误差为：', mean_squared_error(std_y.inverse_transform(y_test), std_y.inverse_transform(y_pred2)))

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()

'''
本文亮点：
①各列数据的控制情况查看：df.isnull().sum()
②删除有空值的行数据：df.dropna(axis=0, inpalce=True)
③回归模型的基本评价指标：MSE；
  分类模型的基本评价指标：精确率
'''







