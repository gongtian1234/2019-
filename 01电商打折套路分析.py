# 导入模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from bokeh.plotting import figure,show,output_file
from bokeh.models import ColumnDataSource

# (1)导入数据
import os
os.chdir('E:/python/实战/网易云数据分析实战/01电商打折套路解析')
df = pd.read_excel('双十一淘宝美妆数据.xlsx')
# print(df.head())
df.fillna(0, inplace=True)
# print(df['update_time'])
df.index = df['update_time']
# print(df.index)
df['date'] = df.index.day    # 加载数据，提取销售日期
# 商品总数和品牌总数
print('商品总数为：' + str(len(df['id'].unique())))
print('品牌总数为：' + str(len(df['店名'].unique())))
# (2)双十一在售商品的占比情况
data1 = df[['id', 'title', '店名', 'date']]    # 筛选数据
d1 = data1[['id', 'date']].groupby(by='id').agg(['min', 'max'])    # 统计不同商品的销售开始和结束时间
id_11 = data1[data1['date']==11]['id']    # 把在11日当天的id找出来
d2 = pd.DataFrame({'id': id_11, '双十一当天是否售卖': True})
id_data = pd.merge(d1, d2, left_index=True, right_on='id', how='left')    # 合并数据
id_data.fillna(False, inplace=True)
m = len(d1)
m_11 = len(id_11)
m_pre = m_11/m
print('双十一当天参与活动的商品为%i个，占比为%.2f%%'%(m_11, m_pre*100))

# (3)商品销售节奏分类，可分为7类
id_data['type'] = '待分类'
id_data['type'][(id_data[('date', 'min')]<11) & (id_data[('date', 'max')]>11)] = 'A'     # 双十一前后及当天都在销售
id_data['type'][(id_data[('date', 'min')]<11) & (id_data[('date', 'max')]==11)] = 'B'    # 双十一前及当天在销售
id_data['type'][(id_data[('date', 'min')]==11) & (id_data[('date', 'max')]>11)] = 'C'    # 双十一后及当天都在销售
id_data['type'][(id_data[('date', 'min')]==11) & (id_data[('date', 'max')]==11)] = 'D'   # 仅在双十一当天销售
id_data['type'][id_data['双十一当天是否售卖']==False] = 'F'                              # 双十一当天未销售，但前后有销售
id_data['type'][id_data[('date', 'max')]<11] = 'E'                                       # 仅在双十一前销售
id_data['type'][id_data[('date', 'min')]>11] = 'G'                                       # 仅在双十一后销售
result1 = id_data['type'].value_counts()    # 计算不同类别商品的数量
from bokeh.palettes import brewer
colori = brewer['YlGn'][7]
plt.axis('equal')
plt.pie(result1, labels=result1.index, autopct='%.2f%%', colors=colori, startangle=90, radius=1.5, counterclock=False)
plt.show()

# (4)未参与双十一当天活动的商品，其去向又如何
'''
## 有四种情况：①暂时下架(F); 
## ②重新上架(E中部分数据，同一个id，有不同的title，换个马甲重新上架)
## ③预售(E中部分数据，预售商品的title中含有“预售”二字)
## ④彻底下架(E中部分数据，可忽略)
'''
## 从中筛选出这几种情况
id_not11 = id_data[id_data['双十一当天是否售卖']==False]
df_not11 = id_not11[['id', 'type']]
# 找到双十一当天未参与活动的商品对应的原始数据
data_not11 = pd.merge(df_not11, df, on='id', how='left')
### ①暂时下架
id_con1 = id_data['id'][id_data['type']=='F'].values
### ②重新上架(E中部分数据，同一个id，有不同的title，换个马甲重新上架)
data_con2 = data_not11[['id', 'title', 'date']].groupby(by=['id', 'title']).count()
#### 同一id，不同title的计数
title_count = data_con2.reset_index()['id'].value_counts()
id_con2 = title_count[title_count>1].index
### ③预售(E中部分数据，预售商品的title中含有“预售”二字)
data_con3 = data_not11[data_not11['title'].str.contains('预售')]
id_con3 = data_con3['id'].value_counts().index
### ④彻底下架(E中部分数据，可忽略)

# (5)真正参与双十一活动的商品（=当天在售的+预售商品）及品牌情况
data_11sale = id_11
id_11sale_final = np.hstack((data_11sale, id_con3))    # 竖向合并
result2_i = pd.DataFrame({'id': id_11sale_final})    # 得到真正参与双十一活动的商品
x1 = pd.DataFrame({'id': data_11sale})
x1_df = pd.merge(x1, df, on='id', how='left')
brand_11sale = x1_df.groupby('店名')['id'].count()    # 不同品牌双十一当天在售的商品数量
x2 = pd.DataFrame({'id': id_con3})
x2_df = pd.merge(x2, df, on='id', how='left')
brand_ys = x2_df.groupby('店名')['id'].count()    # 不同品牌的预售商品数量
result2_data = pd.DataFrame({'当天参与活动的商品数量': brand_11sale,
                             '预售商品数量': brand_ys
                            })
result2_data['总量'] = result2_data['当天参与活动的商品数量'] + result2_data['预售商品数量']
result2_data.sort_values(by='总量', inplace=True, ascending=False)    # 计算结果

# (6)堆叠图制作
from bokeh.models import HoverTool
from bokeh.core.properties import value    # 堆叠图里做属性计算的
lst_brand = result2_data.index.tolist()    # 横坐标
lst_type = result2_data.columns.tolist()[:2]
color = ['red', 'green']    # 基本参数设置
result2_data.index.name = 'brand'
result2_data.columns = ['sale_on_11', 'presale', 'sum']
source = ColumnDataSource(result2_data)
hover = HoverTool(tooltips=[('品牌', '@brand'), ('双十一当天参与活动的数量', '@sale_on_11'),
                            ('预售商品数量', '@presale'),
                            ('真正参与双十一活动的商品总数', '@sum')
                            ])
output_file('project08.html')
p = figure(x_range=lst_brand, plot_width=900, plot_height=350,
           title='各个品牌参与双十一活动的情况',
           tools=[hover, 'box_select, pan,reset,wheel_zoom,crosshair']
           )
p.vbar_stack(lst_type,x='brand',source=source,
             width=0.5, color=color, alpha=0.7,
             legend=[value(x) for x in lst_type],
             muted_color='block',muted_alpha=0.2)
show(p)

# 2、是否真的打折；打折率是多少
data2 = df[['id', 'price', 'title', '店名', 'date']]
data2['period'] = pd.cut(data2['date'], [4,10,11,15], labels=['双十一前','双十一当天','双十一后'])
price = data2[['id', 'price', 'period']].groupby(['id','period']).min()
price.reset_index(inplace=True)
price.dropna(inplace=True)
## 查看价格是否有波动
id_count = price['id'].value_counts()
id_type1 = id_count[id_count==1].index    # 不打折的，有943个
id_type2 = id_count[id_count!=1].index    # 打折的，有2559个
# (2)针对在打折的商品，其折扣率是多少
result3_data2 = data2[['id','price','period','店名']].groupby(['id','period']).min()
result3_data2.reset_index(inplace=True)
result3_data2.dropna(inplace=True)
result3_before11 = result3_data2[result3_data2['period']=='双十一前']
result3_at11 = result3_data2[result3_data2['period']=='双十一当天']
result3_data3 = pd.merge(result3_before11, result3_at11, on='id')
## 计算折扣率=（原价-现价）/ 原价
result3_data3['zkl'] = (result3_data3['price_x']-result3_data3['price_y'])/result3_data3['price_x']
bokeh_data1 = result3_data3[['id', 'zkl']].dropna()
bokeh_data1['zkl_range'] = pd.cut(bokeh_data1['zkl'], bins=np.linspace(0,1,21))
bokeh_data2 = bokeh_data1.groupby('zkl_range').count()
## 再删除折扣率较小的商品[0,0.05]
bokeh_data2 = bokeh_data2.iloc[1:]
bokeh_data2['zkl_pre'] = bokeh_data2['zkl']/bokeh_data2['zkl'].sum()    # 折扣区间的占比
# 开始绘图
output_file('project082.html')
source1 = ColumnDataSource(bokeh_data2)
lst_zkl = bokeh_data2.index.tolist()
hover = HoverTool(tooltips=[('折扣率','@zkl')])
try:
    p = figure(x_range=lst_zkl, plot_width=900, plot_height=350,
               title='商品折扣率统计', tools=[hover,'reset,wheel_zoom,pan,crosshair'])
    p.line(x='zkl_range', y='zkl_pre', source=source1, line_color='block', line_dash=[10,4])
    p.circle(x='zkl_range',y='zklpre', source=source1, size=8, color='red', alpha=0.8)
    show(p)
except:
    print('"figure(x_range=lst_zkl"里的lst_zkl报错，暂时无法解决')
# (3)按照品牌分析不同品牌的打折力度
from bokeh.transform import jitter
brand = result3_data3['店名_x'].dropna().unique().tolist()
bokeh_data3 = result3_data3[['id','zkl','店名_x']].dropna()
bokeh_data3 = bokeh_data3[bokeh_data3['zkl']>0.04]
source2 = ColumnDataSource(bokeh_data3)
output_file('project083.html')
p2 = figure(x_range=brand, plot_width=900, plot_height=700,
               title='不同商品的折扣率', tools=[hover,'box_select, pan,reset,wheel_zoom,crosshair'])
p2.circle(x='zkl',y=jitter('店名_x', width=0.7, range=p2.y_range), source=source2, size=8, color='red', alpha=0.8)
show(p2)

# 3、商家营销套路挖掘
# (1)数据计算，先把折扣的数据计算出来
data_zk = result3_data3[result3_data3['zkl']>0.04]
result4_zkld = data_zk.groupby('店名_x')['zkl'].mean()    # 计算不同品牌的平均折扣率
n_dz = data_zk['店名_x'].value_counts()    # 每个店铺参与打折的商品数
n_bdz = result3_data3['店名_x'].value_counts()
result4_dzspbl = pd.DataFrame({'打折商品数':n_dz, '不打折商品数': n_bdz})
result4_dzspbl['参与打折商品比例'] = result4_dzspbl['打折商品数']/(result4_dzspbl['打折商品数']+result4_dzspbl['不打折商品数'])
result4_dzspbl.dropna(inplace=True)
result4_sum = result2_data.copy()
result4_data = pd.merge(pd.DataFrame(result4_zkld), result4_dzspbl, left_index=True, right_index=True, how='inner')
result4_data = pd.merge(result4_data, result4_sum, left_index=True, right_index=True, how='inner')
# (2)绘制散点图
from bokeh.models.annotations import Span, Label, BoxAnnotation
bokeh_data = result4_data[['zkl', 'sum', '参与打折商品比例']]
bokeh_data.columns = ['zkl', 'amount', 'pre']    # 更改列名
bokeh_data['size'] = bokeh_data['amount']*0.03   # 进行放缩
source = ColumnDataSource(bokeh_data)
##==================================================================================##
x_mean = bokeh_data['pre'].mean()    # 两条线
y_mean = bokeh_data['zkl'].mean()
hover = HoverTool(tooltips=[('品牌', '@index'),
                            ('折扣率', '@zkl'),
                            ('商品总数', '@amount'),
                            ('参与打折商品比例', '@pre')])
output_file('project084.html')
p = figure(plot_width=600, plot_height=600,
           title='各个品牌打折套路解析',
           x_axis_label='参与打折商品比例',
           y_axis_label='折扣率',
           tools=[hover,'box_select, pan,reset,wheel_zoom,crosshair'])
p.circle_x(x='pre', y='zkl', source=source, size='size',
           fill_color='red', line_color='black', fill_alpha=0.6, line_dash=[8,3])
p.ygrid.grid_line_dash = [6,4]
p.xgrid.grid_line_dash = [6,4]
x = Span(location=x_mean, dimension='height', line_color='green', line_dash=[6,4],
         line_alpha=0.5)
y = Span(location=y_mean, dimension='width', line_color='green', line_dash=[6,4],
         line_alpha=0.5)
p.add_layout(x)
p.add_layout(y)
### 给每一个象限内写字
bg1 = BoxAnnotation(bottom=y_mean, right=x_mean, fill_alpha=0.1,fill_color='olive')
label1 = Label(x=0.1, y=0.55, text='少量大打折')
p.add_layout(bg1)
p.add_layout(label1)
bg2 = BoxAnnotation(bottom=y_mean, left=x_mean, fill_alpha=0.1,fill_color='olive')
label2 = Label(x=0.35, y=0.55, text='大量大打折')
p.add_layout(bg2)
p.add_layout(label2)
bg3 = BoxAnnotation(top=y_mean, right=x_mean, fill_alpha=0.1,fill_color='olive')
label3 = Label(x=0.1, y=0.2, text='少量少打折')
p.add_layout(bg3)
p.add_layout(label3)
bg4 = BoxAnnotation(bottom=y_mean, right=x_mean, fill_alpha=0.1,fill_color='olive')
label4 = Label(x=0.35, y=0.2, text='大量少打折')
p.add_layout(bg4)
p.add_layout(label4)

show(p)


