# coding=utf-8

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']='sans-serif'

os.chdir('E:/python/实战/网易云数据分析实战/02知乎数据清洗整理和结论研究')
df = pd.read_csv('test2.csv', encoding='utf-8')
df2 = pd.read_csv('city_people.csv', encoding='utf-8')    # [文件名中最好不要有中文]

# 1、创建函数，用fillna填充缺失数据
def data_fillna(df):
    columns_lst = df.columns
    for col in columns_lst:
        if df[col].dtype=='object':
            df[col].fillna('缺失数据', inplace=True)
        else:
            df[col].fillna(0, inplace=True)
    return df
df = data_fillna(df)
df2.dropna(inplace=True)

# 2、知友全国地域分布情况，分析出TOP20
# 要求：
# ① 按照地域统计 知友数量、知友密度（知友数量/城市常住人口），不要求创建函数
# ② 知友数量，知友密度，标准化处理，取值0-100，要求创建函数
# ③ 通过多系列柱状图，做图表可视化
result1_d1 = df[['id', '居住地']].groupby('居住地').count()
df2['city'] = df2['地区'].str[:-1]    # 城市等级清洗，去掉“市”“省”
result1_d2 = df2[['city', '常住人口']]
result1_data = pd.merge(result1_d1, result1_d2, left_index=True, right_on='city', how='inner')
result1_data['density'] = result1_data['id']/result1_data['常住人口']
# (2)将数据极值化
def data_std(df, *cols):
    for col in cols:
        df[col + '_std'] = (df[col]-df[col].min())/(df[col].max()-df[col].min()) * 100
    return df
result1_data = data_std(result1_data, 'id', 'density')
result1_zysl = result1_data.sort_values(by='id_std', ascending=False)    # zysl:知友数量
result1_zymd = result1_data.sort_values(by='density_std', ascending=False)    # zymd:知友密度
# (3)作图
fig1 = plt.figure(figsize=(16,9))
y1 = result1_zysl['id_std'].iloc[:20]
plt.bar(np.arange(20), round(y1,1), width=0.8, facecolor='yellowgreen', edgecolor='k',
        tick_label=result1_zysl['city'].iloc[:20])
plt.grid(True, linestyle='--', color='gray', linewidth=0.5, axis='y')
plt.title('知友数量TOP20')
for i, j in zip(range(20), y1):
    plt.text(i-0.1, 2, '%.1f'%j, color='k', fontsize=9)
fig2 = plt.figure(figsize=(16,9))
y2 = result1_zymd['density_std'].iloc[:20]
plt.bar(np.arange(20), round(y2,1), width=0.8, facecolor='yellowgreen', edgecolor='k',
        tick_label=result1_zymd['city'].iloc[:20])
plt.grid(True, linestyle='--', color='gray', linewidth=0.5, axis='y')
plt.title('知友密度TOP20')
for i, j in zip(range(20), y2):    # 为图标添加标注
    plt.text(i-0.1, 2, '%.1f'%j, color='k', fontsize=9)
plt.show()

# 3、知友全国地域分布情况
# 要求：
# ① 按照学校（教育经历字段） 统计粉丝数（‘关注者’）、关注人数（‘关注’），并筛选出粉丝数TOP20的学校，不要求创建函数
# ② 通过散点图 → 横坐标为关注人数，纵坐标为粉丝数，做图表可视化
# ③ 散点图中，标记出平均关注人数（x参考线），平均粉丝数（y参考线）

result2_df = df[['教育经历', '关注', '关注者', 'id']]
result2_df.groupby('教育经历').count()

result2_df = df[['教育经历', '关注', '关注者', 'id']]
a0 = result2_df.groupby('教育经历').sum()
a1 = a0.sort_values('关注者', ascending=False)
a1 = a0.sort_values('关注', ascending=False)
a1 = a1.drop(['缺失数据','本科','大学','大学本科'])
x_mean = a1['关注'].iloc[:20].mean()
y_mean = a1['关注者'].iloc[:20].mean()
x0 = a1['关注'].iloc[:20]
y0 = a1['关注者'].iloc[:20]
plt.scatter(x0, y0, marker='.', s=y0 / 1000, cmap='Blues', c=y0, alpha=0.5, label='学校')
plt.axvline(x_mean, hold=None, label='平均关注人数:%i人' % x_mean, color='r', linestyle='--', alpha=0.8)    # 绘制x轴参考线
plt.axhline(y_mean, hold=None, label='平均关注者人数:%i人' % y_mean, color='g', linestyle='--', alpha=0.8)
plt.legend(loc='upper left')
plt.grid()
for i, j, n in zip(x0, y0, a1.index[:20]):
    plt.text(i + 500, j, n, color='k')
plt.show()
