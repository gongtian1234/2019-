# coding=utf-8
import pandas as pd
from datetime import date
import numpy as np
import pickle

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve


# 1、读取数据
## dftest为用户O2O线下优惠券使用预测样本
f1 = open('E:/python/实战/机器学习实战/Cp2_KNN/ofo_knn/ofo_knn/Code/data/ccf_offline_stage1_test_revised.csv', encoding='utf-8')
dftest = pd.read_csv(f1, keep_default_na=False)
f1.close()
## dfoff为用户线下消费和优惠券领取行为
f2 = open('E:/python/实战/机器学习实战/Cp2_KNN/ofo_knn/ofo_knn/Code/data/ccf_offline_stage1_train.csv')
dfoff = pd.read_csv(f2, keep_default_na=False)
## dfon为用户线上点击/消费和优惠券领取行为
f3 = open('E:/python/实战/机器学习实战/Cp2_KNN/ofo_knn/ofo_knn/Code/data/ccf_online_stage1_train.csv')
dfon = pd.read_csv(f3, keep_default_na=False)
# 简单统计线下用户使用优惠券的情况
print('无优惠券，有消费：' + str(dfoff[(dfoff['Date_received']=='null') &
                             (dfoff['Date']!='null')].shape[0]))
print('无优惠券，无消费：' + str(dfoff[(dfoff['Date_received']=='null') &
                             (dfoff['Date']=='null')].shape[0]))
print('有优惠券，无消费' + str(dfoff[(dfoff['Date_received']!='null') &
                             (dfoff['Date']=='null')].shape[0]))
print('有优惠券，有消费' + str(dfoff[(dfoff['Date_received']!='null') &
                             (dfoff['Date']!='null')].shape[0]))
## 分析发现，大量的优惠券投放是无效的，有消费的人群无券，无消费的人群却有券

# 2、特征提取
## 打折率处理
print('打折率类型：' + str(dfoff['Discount_rate'].unique()))
### 打折率分为三种情况：null没有打折；-:-满多少减多少；[0,1]打折率
### 处理方式：打折类型getDiscountType(); 折扣率convertRate(); 满多少getDiscountMan；减多少getDiscountJian
def getDiscountType(row):
    if row=='null':
        return 'null'
    elif ':' in row:    # 表示满减类型
        return 1
    else:
        return 0

def convertRate(row):
    if row=='null':
        return 'null'
    elif ':' in row:    # 全部转换为打折率
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def getDiscountMan(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getDiscountJian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0

def processData(df):
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)   # 打折类型
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    return df

### 对数据进行处理
dfoff = processData(dfoff)
dftest = processData(dftest)

## (2)距离处理
print('distance_type: ' + str(dfoff['Distance'].unique()))
### 把null处理为-1
dfoff['distance'] = dfoff['Distance'].replace('null', -1).astype(int)
dftest['distance'] = dftest['Distance'].replace('null', -1).astype(int)
print('dfoff的distance类型：' + str(dfoff['distance'].unique()))
print('dftest的distance类型：' + str(dftest['distance'].unique()))

## (3)领券日期data_received（到目前为止已经构造出5个特征：discount_type\discount_rate\discount_man\discount_jian\distance）
### 查看一下数据
date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[date_received!='null'])
print('优惠券收到日期从',date_received[0],'到',date_received[-1])
date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[date_buy!='null'])
print('消费日期从',date_buy[0],'到',date_buy[-1])
### 特征构造
'''
关于领劵日期的特征：
weekday : {null, 1, 2, 3, 4, 5, 6, 7} （表示星期几领取的优惠券）
weekday_type : {1, 0}（周六和周日为1，其他为0） （周末还是工作日领取的优惠券）
Weekday_1 : {1, 0, 0, 0, 0, 0, 0}
Weekday_2 : {0, 1, 0, 0, 0, 0, 0}
Weekday_3 : {0, 0, 1, 0, 0, 0, 0}
Weekday_4 : {0, 0, 0, 1, 0, 0, 0}
Weekday_5 : {0, 0, 0, 0, 1, 0, 0}
Weekday_6 : {0, 0, 0, 0, 0, 1, 0}
Weekday_7 : {0, 0, 0, 0, 0, 0, 1}
'''
def getWeekday(row):    # 得到优惠券的星期提取
    if row=='null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:])).weekday() + 1
dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)
### weekday_type: 周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x:1 if x in [6,7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x:1 if x in [6,7] else 0)
### 将weekday转换为onehot编码(用pandas)
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
tmpdf = pd.get_dummies(dfoff['weekday'].replace('null',np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf
tmpdf = pd.get_dummies(dftest['weekday'].replace('null',np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf

## 此时已经构造出所有的特征，有14个：discount_type\discount_rate\discount_man\discount_jian\distance\weekday\weekday_type\ ##
## weekday_1-7 ##

# 3、标签标注，因为原始数据中没有给出，所以需要自己进行标注
'''
三种情况：
①Data_received=='null'，没有领导优惠券，无需考虑，y=-1
②(Data_received!='null')&(Date!='null')&(Date-Date_received<15):表示领取优惠券且在15天内使用，即正样本y=1
③(Data_received!='null')&(Date!='null')&(Date-Date_received>15):表示领取优惠券但未在15天内使用，即正样本y=0
'''
def label(row):
    if row['Date_received']=='null':
        return -1
    if row['Date']!='null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td<pd.Timedelta(15,'D'):
            return 1
    return 0
dfoff['label'] = dfoff.apply(label,axis=1)
print(dfoff['label'].value_counts())

# 4、训练模型——SGDClassifier(随机梯度下降)
## (1)划分训练集\验证集
df = dfoff[dfoff['label']!=-1]
train = df[df['Date_received']<'20160516']
vaild = df[(df['Date_received']>='20160516') & (df['Date_received']<='20160615')]
print('train set:', train['label'].value_counts())
print('Valid set:', vaild['label'].value_counts())
## (2)特征数量
original_feature = ['discount_type', 'discount_rate', 'discount_man', 'discount_jian', 'distance', 'weekday', 'weekday_type',
                    ] + weekdaycols
print('共有{}个特征'.format(len(original_feature)))
print(original_feature)
## (3)建立模型
def check_model(data, predictors):
    classifier = lambda: SGDClassifier(
        loss='log',    # loss function:logistic regression逻辑回归损失函数
        penalty='elasticnet',    # L1&L2, 'elasticnet'是一个弹性的正则化，将L1和L2进行了结合
        fit_intercept=True,    # 是否存在截距，默认存在
        max_iter=100,
        shuffle=True,    # Whether or not the training data should be shuffled after each epoch(是否在对训练数据完成训练后，进行随机打乱)
        n_jobs=1,    # the number of processors to use
        class_weight=None,    # Weights associated with classes.(每个类别的权重)If not given, all classes are supposed to have weight one.
    )
    # 管道机制使得参数集在新数据集上的重复使用，管道机制实现了对全部步骤的流式化封装和管理
    model = Pipeline(steps=[
        ('ss', StandardScaler()),    # transformer
        ('en', classifier())    # estimator
    ])
    parameters = {
        'en__alpha': [0.001,0.01,0.1],
        'en__l1_ratio': [0.001,0.01,0.1]
    }
    # StratifiedKFold与KFold类似，但它是分层采样，确保训练集、测试集中各类别样本的比例与原始数据集中的相同
    folder = StratifiedKFold(n_splits=3, shuffle=True)    # n_splits:Number of folds
    ## 交叉验证筛选参数
    grid_search = GridSearchCV(
        model,
        param_grid=parameters,
        cv=folder,
        n_jobs=2,    # -1 means using all processors
        verbose=1    # Controls the verbosity: the higher, the more messages.
    )
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])
    return grid_search

## (4)训练
predictors = original_feature
model = check_model(train, predictors)

## (5)验证
### 对验证集中每个优惠券预测的结果计算AUC，再对所有优惠券的AUC求平均。计算AUC的时候，如果label只有一类，就直接跳过，因为AUC无法计算。
# valid predict
y_valid_pre = model.predict_proba(vaild[predictors])
valid1 = vaild.copy()
valid1['pred_prob'] = y_valid_pre[:,1]
# 计算AUC
vg = valid1.groupby(['Coupon_id'])
aucs = []
for i in vg:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

## (6)测试
# test prediction for submission
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['Probability'] = y_test_pred[:,1]
dftest1.to_csv('E:/python/实战/机器学习实战/Cp2_KNN/ofo_knn/ofo_knn/Code/data/submit1.csv', index=False, header=False)

## (7)保存模型
import os
os.chdir('E:/python/实战/机器学习实战/Cp2_KNN/ofo_knn/ofo_knn/Code/data/')
if not os.path.isfile('1_model.pkl'):
    with open('1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('1_model.pkl', 'rb') as f:
        model = pickle.load(f)

# https://nbviewer.jupyter.org/github/gongtian1234/Machine-Learning-in-Action/blob/master/Cp2_KNN/ofo_knn/o2o-1.ipynb
'''
本文亮点内容：
①读取数据时，空值不显示为'NaN', 显示为'null'
read_csv('',keep_default_na=False)
文中为：读取数据时
②调用函数的另一种方式：用apply
data[row].astype(str\int).apply(getWeekday)
文中为：函数后的调用在本文都采用这种方式
③从年月日的日期中找到对应的星期几：
from datetime import date
date(2019,7,22).weekday() + 1    # 默认周一对应的是0
文中为：date(int(row[0:4]), int(row[4:6]), int(row[6:])).weekday() + 1
④是周末则赋值为1，否则为0；
文中为：dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x:1 if x in [6,7] else 0)
⑤用pandas的get_dummies()将某一列转换为one-hot编码, 并附上自定义的列名
文中为:
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
tmpdf = pd.get_dummies(dfoff['weekday'].replace('null',np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf
⑥两个相邻的日期求间隔天数：
pd.to_datetime('20190722',format='%Y%m%d')
文中为：
td = pd.to_datetime(dfoff['Date'],format='%Y%m%d')-pd.to_datetime(dfoff['Date_received'],format='%Y%m%d')
⑦垃圾清理
import gc
gc.collect()
⑧统计某一列各个值出现的次数
文中为：print(dfoff['label'].value_counts())
⑨SGDClassifier(随机梯度下降)模型的使用：整个过程
文中为：第四部分，模型部分
'''


