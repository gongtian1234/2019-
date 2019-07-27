# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
from scipy import stats
import random

df = pd.read_excel('moviedata.xlsx')
# 1、以‘豆瓣评分’为标准查看电影分布及烂片情况
## (1)查看豆瓣评分数据分布，绘制直方图和箱型图
df_score = df[(df['豆瓣评分'].notnull()) & (df['豆瓣评分']>0)]['豆瓣评分']
print('豆瓣评分数据的基本情况：' + '\n', df_score.describe())    # 从这可以找到其四分位数
plt.hist(df_score, bins=20, facecolor='green', alpha=0.5)
plt.grid(axis='y', linestyle='--', c='gray')
plt.show()
# df_score.plot.box(vert=False, grid=True, figsize=(10,4))
# plt.show()
plt.boxplot(df_score, vert=False, meanline=True, showmeans=True, patch_artist=True)
score_info = df_score.describe()    # 类型为：pandas.core.series.Series是一个表
df_score_info = [score_info['min'], score_info['25%'], score_info['50%'], score_info['75%'], score_info['max']]
for i, j, n in zip(df_score_info, [1]*5, df_score_info):
    plt.text(i, j, n, color='k')
plt.show()
### 下四分位数对应的值为4.3，所以认为小于4.3的就是烂片
# (2)用ks检验豆瓣评分数据是否符合正态分布
u = df_score.mean()
std = df_score.std()
print('豆瓣评分的正态性检验（ks检验）的p值为：' + '\n', stats.kstest(df_score, 'norm', (u, std)).pvalue)
### 检验结果表明其不是正态分布
# (3)筛选出烂片数据，并做排名，找到TOP20
data_lp = df[(df['豆瓣评分']<4.3) & (df['豆瓣评分']>0)]
lp_top20 = data_lp[['电影名称', '豆瓣评分', '主演', '导演']].sort_values(by='豆瓣评分', ascending=True).iloc[:20].reset_index()
del lp_top20['index']

# 2、什么题材的电影烂片最多
## (1)按照类型字段分类，筛选不同电影属于什么题材
### 先得到所有的电影类型
typelist = []
for i in df[df['类型'].notnull()]['类型'].str.replace(' ', '').str.split('/'):
    typelist.extend(i)
typelist = list(set(typelist))
### 再去统计不同类型的烂片率
df_type = df[df['类型'].notnull()]
def f1(data, typei):
    dic_type_lp = {}
    datai = data[data['类型'].str.contains(typei)]
    dic_type_lp['typename'] = typei
    dic_type_lp['typecount'] = len(datai)
    dic_type_lp['type_lp_pre'] = len(datai[datai['豆瓣评分']<4.3])/len(datai)
    return dic_type_lp
lst_type_lp = []
for i in typelist:
    lst_type_lp.append(f1(df_type, i))
df_type_lp = pd.DataFrame(lst_type_lp)
type_lp_top20 = df_type_lp.sort_values(by='type_lp_pre', ascending=False).iloc[:20]
plt.scatter(type_lp_top20['typecount'], type_lp_top20['type_lp_pre'], s=type_lp_top20['typecount'], c='green', alpha=0.8)
plt.ylabel('type_lp_pre')
plt.xlabel('typecount')
type_lp_top20 = type_lp_top20.sort_values(by='typecount', ascending=False)
for i,j,n in zip(type_lp_top20['typecount'], type_lp_top20['type_lp_pre'], type_lp_top20['typename']):
    plt.text(i,j,n,color='gray')
plt.show()

# 3、与那些国家合拍更容易产生烂片
## 按照“制片国家\地区”分类，筛选出不同电影的制片地
df_loc = df[['电影名称', '豆瓣评分', '制片国家/地区']][df['制片国家/地区'].notnull()]
### 有些没有中国参与的，直接删除
# df_loc = df_loc[df_loc['制片国家/地区'].str.contains('中国大陆')]   # 这会损失掉只有香港或台湾的数据
df_loc['是否国产'] = None
for i in range(len(df_loc['制片国家/地区'])):
    if '中国大陆' in df_loc['制片国家/地区'][i] or '香港' in df_loc['制片国家/地区'][i] or '台湾' in df_loc['制片国家/地区'][i]:
        df_loc['是否国产'][i] = True
    else:
        df_loc['是否国产'][i] = False
df_loc = df_loc[df_loc['是否国产']==True]
###
loclst = []
for i in df_loc['制片国家/地区'].str.replace(' ', '').str.split('/'):
    loclst.extend(i)
loclst = list(set(loclst))
loclst.remove('香港')
loclst.remove('台湾')
loclst.remove('中国大陆')
loclst.remove('中国香港')
loclst.remove('中国')
def f2(data, loci):
    dic_loc_lp = {}
    datai = data[data['制片国家/地区'].str.contains(loci)]
    dic_loc_lp['loc'] = loci
    dic_loc_lp['loccount'] = len(datai)
    dic_loc_lp['loc_lp_pre'] = len(datai[datai['豆瓣评分']<4.3])/len(datai)
    return dic_loc_lp
lst_loc_lp = []
for i in loclst:
    lst_loc_lp.append(f2(df_loc, i))
df_loc_lp = pd.DataFrame(lst_loc_lp)
df_loc_lp = df_loc_lp[df_loc_lp['loccount']>=3]
loc_lp_top20 = df_loc_lp.sort_values(by='loccount', ascending=False).iloc[:20]

# 4、主演人数是否与烂片有关
df['主演人数'] = df['主演'].str.replace(' ', '').str.split('/').str.len()
df_loadrole1 = df[['主演人数', '豆瓣评分']].groupby('主演人数').count()
df_loadrole2 = df[['主演人数', '豆瓣评分']][df['豆瓣评分']<4.3].groupby('主演人数').count()
df_lead_role = pd.merge(df_loadrole1, df_loadrole2, left_index=True, right_index=True)
df_lead_role.columns = ['电影数量', '烂片数量']
df_lead_role.reset_index(inplace=True)
df_lead_role['主演人数分类'] = pd.cut(df_lead_role['主演人数'], [0,2,4,6,9,50],
                                labels=['1-2人', '3-4人', '5-6人', '7-9人', '10人及以上'])
df_lead_role2 = df_lead_role[['电影数量', '烂片数量', '主演人数分类']].groupby('主演人数分类').sum()
df_lead_role2['烂片比例'] = df_lead_role2['烂片数量'] / df_lead_role2['电影数量']

# 4、不同导演每年的电影产量
## (1)通过“上映日期”筛选每个电影的上映年份
df_syrq = df[df['上映日期'].notnull()]
df_syrq['year'] = df_syrq['上映日期'].str.replace(' ', '').str[:4]
df_syrq = df_syrq[df_syrq['year'].str[0]=='2']    # 会把19几几年的电影误删
df_syrq['year'] = df_syrq['year'].astype(int)
df_syrq = df_syrq[df_syrq['导演'].notnull()]
## (2)查看不同导演的烂片比率，去除拍过10次以下的导演
directorlst = []
for i in df_syrq['导演']:
    directorlst.extend(str(i).replace(' ', '').split('/'))
directorlst = list(set(directorlst))    # 去重
def f3(data, diri):
    directori = {}
    datai = data[data['导演'].str.contains(diri)]
    directori['导演'] = diri
    directori['作品数量'] = len(datai)
    if len(datai)!=0:
        directori['烂片比率'] = len(datai[datai['豆瓣评分']<4.3])/len(datai)
    return directori
df_dir = []
for diri in directorlst:
    df_dir.append(f3(df_syrq, diri))
df_dir = pd.DataFrame(df_dir)
df_dir = df_dir[df_dir['烂片比率'].notnull()]
df_dir2 = df_dir[df_dir['作品数量']>=10]
## (3)查看不同导演每年的电影产量，并制作散点图
def f4(data, diri):
    datai = data[data['导演'].str.contains(diri)]
    data_moivecount = datai[['year', '电影名称']].groupby('year').count()
    data_moivemean = datai[['year', '豆瓣评分']].groupby('year').mean()
    df_i = pd.merge(data_moivecount, data_moivemean, left_index=True, right_index=True)
    df_i.columns = ['count', 'score']
    df_i['size'] = df_i['count']*5
    return df_i
### 先查看王晶的
dir1data = f4(df_syrq, '王晶')
dir2data = f4(df_syrq, '萧锋')
plt.scatter(x=list(dir1data.index), y=dir1data['score'], s=dir1data['size'], label='王晶')
plt.scatter(x=list(dir2data.index), y=dir2data['score'], s=dir2data['size'], label='萧锋')
plt.legend(loc='upper right')
plt.show()

'''
本文亮点：
①箱型图plt.boxplot()
文中为：
plt.boxplot(df_score, vert=False, meanline=True, showmeans=True, patch_artist=True)
# 从.describe()可以找到数据的四分位数
score_info = df_score.describe()
df_score_info = [score_info['min'], score_info['25%'], score_info['50%'], score_info['75%'], score_info['max']]
for i, j, n in zip(df_score_info, [1]*5, df_score_info):
    plt.text(i,j,n,color='b')
plt.show()
②对数据进行正态性检验（用ks检验）:
文中为：
from scripy import stats
u = df_score.mean()
std = df_score.std()
print(stats.kstest(df_score, 'norm', (u, std)))
③散点图可以填入数据以表示点的大小
④针对“ / / / ”来统计1主演人数（其他类似）：
文中为：df['主演人数'] = df['主演'].str.replace(' ', '').str.split('/').str.len()
⑤依据某一列的数据进行分类统计：如1-2人、3-4人……
pd.cut('所依据的列', [0,2,4,6(取右不取左)], labels=['','',''])
文中为：
df_lead_role['主演人数分类'] =pd.cut(df_lead_role['主演人数'], [0,2,4,6,9,50], labels=['',......]) 

'''

