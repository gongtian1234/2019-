import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
os.chdir('E:/python/实战/机器学习实战/Cp2_KNN/')
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1、读取数据
f = open('./data/二手车估价案例/data.csv')
df = pd.read_csv(f)
f.close()
# 2、清洗数据，将type和color转换为one-hot编码
df_type = pd.get_dummies(df['Type'].astype(str)).add_prefix('type_')
df_color = pd.get_dummies(df['Color'].astype(str)).add_prefix('color_')
df = pd.concat([df, df_type, df_color], axis=1)
## (2)查看一下数据的矩阵相关图
matrix = df.corr()
f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(matrix, square=True)
plt.title('Car price variables')
# 3、建立模型
## (1)划分训练集、测试集
x = df[['Construction Year', 'Odometer', 'Days Until MOT']]
y = df['Ask Price'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
## (2)将数据标准化
x_normalizer = StandardScaler()
x_train = x_normalizer.fit_transform(x_train)
x_test = x_normalizer.transform(x_test)
y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)
## (3)训练模型

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(x_train, y_train)
# 预测
y_test_pre = knn.predict(x_test)
print('均方误差为：' + str(mean_squared_error(y_normalizer.inverse_transform(y_test), y_normalizer.inverse_transform(y_test_pre))))

'''
##【注意】超参数搜索分类算法中用的较多，回归算法中暂时未找到办法##
def check_model(x,y):
    classifier = KNeighborsRegressor()
    parameters = {
        'n_neighbors': [i for i in range(1, 9)]
    }
    # StratifiedKFold与KFold类似，但它是分层采样，确保训练集、测试集中各类别样本的比例与原始数据集中的相同
    folder = StratifiedKFold(n_splits=3, shuffle=True)
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=parameters,
        cv=folder,
        n_jobs=2,
    )
    grid_search = grid_search.fit(x,y)    ##【报错位置】##
    return grid_search
model = check_model(x_train, y_train)
'''

'''
本代码亮点：
①转换为one-hot编码后列名的添加：
add_prefix('str_')
文中为：df_type = pd.get_dummies(df['Type'].astype(str)).add_prefix('type_')
②绘制相关矩阵的heatmap
文中为：
import seaborn as sns
matrix = df.corr()
f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(matrix, square=True)
plt.show()
③总结了超参数搜索的基本函数：
文中为:
def check_model(x,y):
    ##以决策树为例##
    classifier = DecisionTreeClassifier(random_state=1)
    parameters = {
        'max_leaf_nodes': list(range(2,100)),    # 参数是决策树分类器中的，以便进行网格超参数搜索
        'min_samples_split': [8,10,15]
    }
    # StratifiedKFold与KFold类似，但它是分层采样，确保训练集、测试集中各类别样本的比例与原始数据集中的相同
    folder = StratifiedKFold(n_splits=3, shuffle=True)
    # Exhaustive search over specified parameter values for an estimator.
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=parameters,
        cv=folder,
        n_jobs=2,
        verbose=1    # Controls the verbosity: the higher, the more messages.
    )
    grid_search = grid_search.fit(x,y)
    return grid_search
model = check_model(x_train,y_train)
##进行预测模型评估等……##

##保存模型##
import os, pickle
if not os.path.isfile('test_model.pkl'):
    with open('test_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('test_model.pkl', 'rb') as f:    ##这个就是读取文件##
    model = pickle.load(f)


'''

