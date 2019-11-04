@[TOC](分析报告)

 #### 一、探索性数据分析

　　在这一部分主要是对训练集的81个features进行了分析，以对数据有一个初步的了解，方便后续进行特征工程和模型的建立。在进行数据探索之前，先了解一下训练集和测试集的样本量，训练集的shape为(1460, 81)  ，测试集的shape为(1459, 80)，测试集比训练集少了“SalePrice”目标值这一列；此外也统计了训练数据的类别和数值特征，代码如下所示：  

    train_dty = train.dtypes
    train_obj_names = list(train_dty[train_dty=='object'].index)    # 43
    train_int_names = list(train_dty[train_dty=='int64'].index)    # 35   
    train_flt_names = list(train_dty[train_dty=='float64'].index)   # 3
    
　　虽然features中的“Id”与目标值“SalePrice”的type是数值型，但是“Id”是唯一标识，“SalePrice”是目标值，所以将它们从features中删除。  
接下来正式进入数据探索性分析，了解数据的缺失情况、连续及类别型特征的数据分布情况以及各个特征之间的相关性等。  
##### 1 查看缺失值
　　查看训练集的缺失值情况发现共有19个特征存在缺失值，其中有5个特征“FireplaceQu”、“Fence”、“Alley”、“MiscFeature”、“PoolQC”缺失值量大于50%，按照从小到大排序如图1所示：  

![avatar](./images/01.png)
<center><font size=2>图1 缺失值的直方图</font></center>  

##### 2 查看目标值的类型及分布
<font size=3>
　　了解了数据特征的缺失值状态之后，接下来分析目标值的类型疾风步。首先，通过head发现目标值“SalePrice”是连续型变量，说明要解决的问题是回归类型的问题，进一步分析其分布状态，如图2、图3、图4所示，可以看出SalePrice的分布与Johnsonsu分布的拟合度最高，同时很明显的可以看出SalePrice不服从正态分布，在图4中与SalePrice的对数后的正态分布拟合较好，所以认为取对数后的SalePrice服从正态分布，即考虑在后面将SalePrice对数化。  
</font>

![avatar](./images/02.png)
<center><font size=2>图2 SalePrice的分布与Johnsonsu分布对比</font></center> 

![avatar](./images/03.png)
<center><font size=2>图3 SalePrice的分布与正态分布对比</font></center> 

![avatar](./images/04.png)
<center><font size=2>图4 SalePrice的分布与对数正态分布对比</font></center> 

##### 3 检验连续型特征分布是否服从正态性分布
<font size=3>
　　类似的，检验连续型变量是否服从正态分布，根据之前提取出的连续型变量用如下代码进行正态性检验，结果得到False表示，没有一个连续型变量的分布服从正态分布；再进一步画出这些连续型变量的分布图，如图5所示，许多变量的分布不规则，考虑在后面对其进行对数变换。

</font>

		test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
		## shapirpo专门做正态检验的模块，当样本数大于时，不再适合用其做正态分布，得到的p值可能不准确
		normal = pd.DataFrame(train[list(train_int_names+train_flt_names)])
		normal = normal.apply(test_normality)  # normal每列都为true，说明每个都拒绝了正态分布的假设
		print(not normal.any())
 
 ![avatar](./images/05.png)
<center><font size=2>图5 连续型特征的分布图</font></center> 

##### 4 对类变形变量进行初步探索
<font size=3>
　　对类别型变量进行探索使用了两种方式，第一种，先画箱线图初步分析一下每一个类别型特征类间的区分度，如下图6所示，可以看到几乎每个特征下的类别间并不完全相同，初步认为类间存在差异。第二种，通过单因素方差分析来分析类间差距，将得到的p值取倒数再取对数（这样做可以使越小的值变得越大，方便后面画图），再按照大小画图如图7所示，可以看出“Neighborhood” 的类间差距最大，依次类推，而且这些类别特征用单因素方差分析，其p值均小于0.05，可以认为这些类别特征对“SalePrice” 的影响是显著的。  
</font>

![avatar](./images/06.png)
<center><font size=2>图6 类别型特征的箱线图</font></center> 

![avatar](./images/07.png)
<center><font size=2>图7 类别型特征显著性排序图</font></center>

<font size=3>
　　其次，对类别型变量进行one-hot编码，对每一个特征的类别先计数，再根据计数的大小排序，数目最多的类别编码为1，第二多的编码为2，以此往后类推，为了区分编码后的特征，在特征名后面加上“_E”。  

　　最后，经过上面一些列的数据探索分析，了解了数据的缺失值情况，目标值、连续型、离散型特征的分布等情况，接下来正式进入特征工程，为建模做准备。
</font>

#### 二、特征工程
##### 1 数据预处理
###### 01 构造训练集、测试集的features及label
<font size=3>
　　第一步，先将train、test中的“Id”删掉，因为“Id”是具有唯一标识的特征，对研究“SalePrice”来说没有实际意义，所以将其删除；第二步，将train数据集中的“SalePrice”进行对数变换，将其赋值给y，进行对数变换可以使其近似服从正态分布，因为一些模型如线性模型、SVM等要求目标变量服从正态分布，然后将“SalePrice”这一特征从train中删除，此时train、test的特征数都是79个，所以可以将其合并为一张表格一起进行接下来的处理。  

    # 删除无关的特征列Id
    train.drop(['Id'], axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    # 将“SalePrice”对数化，并赋值给y
    train.reset_index(drop=True, inplace=True)
    train['SalePrice'] = np.log1p(train['SalePrice'])
    y = train['SalePrice'].reset_index(drop=True)
    train_features = train.drop(['SalePrice'], axis=1)
    # 将train与test数据集合并
    test_features = test
    features = pd.concat([train_features, test_features]).reset_index(drop=True)
</font>

###### 02 缺失值处理
<font size=3>
　　对类别型特征的缺失值补“None”，对连续型特征的缺失值补0。
</font>  

###### 03 连续型特征的偏度处理
<font size=3>
　　计算出连续型特征的偏度系数，挑选出偏度系数大于0.5的特征，对其进行boxcox变换，从而改变数据的正态性。

    numerics_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics2 = []
    for i in features.columns:
        if features[i].dtype in numerics2:
            numerics2.append(i)
    skew_features = features[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)    # 得到含有各个特征的偏度系数表
    # 挑选出偏度系数大于0.5的特征，对其进行boxcox变换
    high_skew = skew_features[skew_features>0.5]
    skew_index = high_skew.index
    for i in skew_index:
        features[i] = boxcox1p(features[i], boxcox_normmax(features[i]+1))
</font>

##### 2 特征构造
<font size=3>
　　基于特征所表达含义，可以利用连续型和离散型特征构造一些简单的特征，以进一步挖掘数据的潜藏特征。

　　第一步，构造数值型特征，如将房屋的地下室面积、一层面积、二层面积加起来就可以得到房屋的总面积等，具体代码如下所示：

     # 构造数值型变量，主要是针对数值型特征进行了处理
    features['YrBltAndRemod'] = features['YearBuilt'] + features['YearRemodAdd']    # 两个年份加起来
    features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']    # 得到房屋的总面积: 地下室+一层+二层
    features['Total_sqr_footage'] = features['BsmtFinSF1'] + features['BsmtFinSF2'] + features['1stFlrSF'] + features['2ndFlrSF']
    features['Total_Bathrooms'] = features['BsmtFullBath'] + features['BsmtHalfBath'] + features['FullBath'] + features['HalfBath']    # 总的浴室个数
    features['Total_porch_sf'] = features['WoodDeckSF'] + features['OpenPorchSF'] + features['EnclosedPorch'] + \
                features['3SsnPorch'] + features['ScreenPorch']    # 总的门廊面积
</font>
<font size=3>
　　第二步，构造离散型特征，在这里主要是基于某一个离散特征，判断该住宅是否含有这一目标，如“PoolArea”，如果有“PoolArea”，则对“hasPool”赋值为1，否则为0，其余的类似，详见下面的代码：

    features['hasPool'] = features['PoolArea'].apply(lambda x:1 if x>0 else 0)    # 一共就只有12家有游泳池
    features['has2ndFloor'] = features['2ndFlrSF'].apply(lambda x:1 if x>0 else 0)  # 有1249户有二层
    features['hasGarage'] = features['GarageArea'].apply(lambda x: 1 if x>0 else 0) # 有2759户有车库，158户没有
    features['hasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x>0 else 0)# 有1497户有壁炉，1420户没有
    features['hasBsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)  # 有2838户有地下室，79户没有
</font>

<font size=3>
　　第三步，在进行最终的建模前，删除类别型特征中某一类别占比超过99.94%的特征，代码如下所示，通过这种方式一共删除了5+3个对y区分布不大的特特征。

    X = final_features.iloc[:len(y), :]           # 拆分出训练集
    X_sub = final_features.iloc[len(y):, :]       # 拆分出测试集 
    # 用这种方式删除一些特征，如果该特征的value_counts[0]的值占比超过99.94%, 则删除
    overfit = []
    for i in X.columns:
        counts = X[i].value_counts()
        zeros = counts.iloc[0]    # 取第0行
        if zeros/len(X)>0.9994:
            overfit.append(i)	
    X.drop(overfit, axis=1, inplace=True)
    X_sub.drop(overfit, axis=1, inplace=True)
</font>


#### 三、建立模型
<font size=3>
　　在这一部分，会应用多个模型，有岭回归模型、lasso模型、弹性网络模型、支持向量回归模型、GBR、lightgbm、xgboost模型；然后再用xgboost作为次级分类器以stacking的方式进行模型融合，这样也得到一个模型。此时一共有8个模型，然后基于这8个模型分别对测试集进行预测，最后再将预测结果按照不同的权重结合起来，进而得出最终的预测结果（将8个模型预测结果结合起来的函数如下所示）。

    def blend_model_predict(X):
        return (0.1*elasticnet_model.predict(X)) + (0.05*lasso_model.predict(X)) + \
               (0.1*ridge_model.predict(X)) + (0.1*svr_model.predict(X)) + \
               (0.1*gbr_model.predict(X)) + (0.15*xgboost_model.predict(X)) + \
               (0.1*lgbm_model.predict(X)) + (0.3*stack_gen_model.predict(np.array(X)))


　　在融合过程中，各个分类器的rmse如表1所示，可以看出融合后的模型效果要优于单个模型的rmse，表明模型融合是有效的。

<center><font size=2>表1 各模型的rmse</font></center>

模型 | rmse
---- | ----
ridge | 0.1096
lasso | 0.1092
elastic net | 0.1092
SVR | 0.1091
gbr | 0.1135
lightgbm | 0.1148
xgboost | 0.1130
stacking | 0.0597
</font>