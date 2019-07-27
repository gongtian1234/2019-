import matplotlib.pyplot as plt
import jieba, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# 1、读取文件并提取出相关的训练和测试原始内容（因为提取的内容还需要进行处理，分词，停用词，词向量）
def read_data(path, is_pos=None):
    # param path: 文件路径
    # param is_pos: 读取的数据是否为positive samples, 可选值有True\False\None
    # return: reviews, labels分别为评论内容列表和标签列表
    reviews, labels = [], []
    # 从文件中提取出评论内容，和其所对应的标签：积极或消极
    f = open(path, 'r', encoding='utf-8')
    df = f.read()
    f.close()
    reviews = re.findall('<review id="\d.*?">(.*?)</review>', df.replace('\n', ''))
    if is_pos:
        labels = [1] * len(reviews)
    elif is_pos is None:
        labels = re.findall('<review.*?label="(\d*?)">', df)    # 正则提取后的元素是str，需要转换为int型
        labels = [int(i) for i in labels]
    elif is_pos is False:
        labels = [0] * len(reviews)
    return reviews, labels

def process_file():
    '''读取训练数据和测试数据并做一些处理'''
    train_pos_file = 'data_sentiment/train.positive.txt'
    train_neg_file = 'data_sentiment/train.negative.txt'
    test_comb_file = 'data_sentiment/test.combined.txt'
    train_pos_cmts, train_pos_lbs = read_data(path=train_pos_file, is_pos=True)
    train_neg_cmts, train_neg_lbs = read_data(path=train_neg_file, is_pos=False)
    test_comments, test_labels = read_data(path=test_comb_file, is_pos=None)
    train_comments = train_pos_cmts + train_neg_cmts
    train_labels = train_pos_lbs + train_neg_lbs
    return train_comments, test_comments, train_labels, test_labels

train_comments, test_comments, train_labels, test_labels = process_file()
print(len(train_comments), '\n', len(test_comments))

# 2、对得到的数据进行处理（分词、停用词、词向量）
def load_stopwords(path):
    '''从外部文件导入停用词'''
    stopwords = set()
    with open(path, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            stopwords.add(line.strip())    # set只能用add往里面添加元素，不能用append
    return stopwords
def clean_non_chinese_symbols(text):
    '''处理文本中的非中文字符'''
    text = re.sub('[!！]', '！', text)
    text = re.sub('[a-zA-Z@#$%^&*()/+=<>。’‘“”，；：:;,."《》…【】~{}]', 'UNK', text)
    return re.sub('\s+', ' ', text)    # + 在正则中表示出现一次或多次
def clean_numbers(text):
    '''处理数字符号'''
    return re.sub('\d+', 'NUM', text)
def process_text(text, stopwords):
    '''文本的预处理过程'''
    text = clean_non_chinese_symbols(text)
    text = clean_numbers(text)
    text = ' '.join([term for term in jieba.cut(text) if term and not term in stopwords])
    return text
path_stopwords = 'data_sentiment\stopwords.txt'
stopwords = load_stopwords(path_stopwords)
train_comments_new = [process_text(comment, stopwords) for comment in train_comments]
test_comments_new = [process_text(comment, stopwords) for comment in test_comments]
print(train_comments_new[0], test_comments_new[0])
## 利用tfidf从文本中提取特征，写到数组里面
tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(train_comments_new)
y_train = train_labels
x_test = tfidf.transform(test_comments_new)
y_test = test_labels

# 3、建立模型
## (1)朴素贝叶斯模型
clf1 = MultinomialNB(alpha=1.0, fit_prior=True)
clf1.fit(x_train, y_train)
y_pred = clf1.predict(x_test)
print('naive_bayes的accuracy为' + str(accuracy_score(y_test, y_pred)))
## (2)KNN模型
clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)
y_pred2 = clf2.predict(x_test)
print('KNN的accuracy为' + str(accuracy_score(y_test, y_pred2)))

##==========================================分割线====================================================================##
'''
import re
f = open('data_sentiment/train.positive.txt', 'r')
df = f.read()
f.close()
labels = re.findall('<review id="(\d.*?)">', df)
reviews = re.findall('<review id="\d.*?">(.*?)</review>', df.replace('\n', ''))
'''





