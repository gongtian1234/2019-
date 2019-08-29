# coding=utf-8
from sklearn.datasets import fetch_lfw_people    # 导入人脸数据
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
# 查看15张人脸
for i in range(15):
    plt.subplot(3, 15, i+1)
    plt.imshow(faces.images[i], cmap='bone')
    plt.xlabel(faces.target_names[faces.target[i]])
plt.show()
## 每一幅图为[62,47]将其平展为一个向量(3000)，然后通过PCA将其降维更短的向量(150)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.3)

# 建立模型
def check_model(x, y):
    '''用于训练模型，所以要传入训练集'''
    pca = PCA(n_components=150,
              whiten=True,    # 白化：由于图像相邻像素间具有很强的相关性，所以训练时输入是冗余的，通过白化消除这种强相关性，并使其具有相同的方差
              )
    svc = SVC(kernel='linear',    # kernel的种类有：linear线性核函数，poly多项式~，rbf高斯核，sigmoid核，precomputed核矩阵
              class_weight='balanced'    # 给每个类别设置惩罚参数，如果没有，则都给1；如果为'balanced'，则使用y的值自动调整与输入数据中类频率成反比的权重
              )
    pipeline = make_pipeline(pca, svc)    # 管道机制
    parameters = {'svc__C': [1, 5, 10, 50]}
    folder = StratifiedKFold(n_splits=5, shuffle=True)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=folder, n_jobs=2, verbose=1)
    gridSearch = grid_search.fit(x, y)
    print(gridSearch.best_params_)
    return gridSearch
model = check_model(x_train, y_train)
model = model.best_estimator_

# 使用训练好的模型做预测
y_pred = model.predict(x_test)
## 作图显示(这个仅适用于对图的预测，鸡肋)
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[y_pred[i]].split()[-1], color='black' if y_pred[i] == y_test[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)

# 评价模型
## 用准确率进行评价
print("model's accuracy is", accuracy_score(y_test, y_pred))
## 生成性能报告
print(classification_report(y_test, y_pred, target_names=faces.target_names))






