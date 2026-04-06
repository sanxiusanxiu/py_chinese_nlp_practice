# 8.4 任务：垃圾短信分类

import os
import re
import jieba
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# 代码8-3 加载库和读取数据
data = pd.read_csv('data/messages.csv', encoding='utf-8', index_col=0, header=None)
data.columns = ['类别', '短信']
# print(data.类别.value_counts())

# 代码8-4 文本预处理
temp = data.短信
temp.isnull().sum()
# 去重
data_dup = temp.drop_duplicates()
# 脱敏
l1 = data_dup.astype('str').apply(lambda x: len(x)).sum()
data_qumin = data_dup.astype('str').apply(lambda x: re.sub('x', '', x))
l2 = data_qumin.astype('str').apply(lambda x: len(x)).sum()
print('脱敏 减少了' + str(l1 - l2) + '个字符')
# 加载自定义词典
jieba.load_userdict('data/newdic1.txt')
# 分词
data_cut = data_qumin.astype('str').apply(lambda x: list(jieba.cut(x)))
# 去停用词
stopword = pd.read_csv('data/stopword.txt', sep='ooo', encoding='gbk', header=None, engine='python')
stopword = [' '] + list(stopword[0])
l3 = data_cut.astype('str').apply(lambda x: len(x)).sum()
data_qustop = data_cut.apply(lambda x: [i for i in x if i not in stopword])
l4 = data_qustop.astype('str').apply(lambda x: len(x)).sum()
print('去停用词 减少了' + str(l3 - l4) + '个字符')
data_qustop = data_qustop.loc[[i for i in data_qustop.index if data_qustop[i] != []]]

# 代码8-4 统计词频并绘制词云图
lab = [data.loc[i, '类别'] for i in data_qustop.index]
lab1 = pd.Series(lab, index=data_qustop.index)


def cipin(data_qustop, num=10):
    temp = [' '.join(x) for x in data_qustop]
    temp1 = ' '.join(temp)
    temp2 = pd.Series(temp1.split()).value_counts()
    return temp2[temp2 > num]


data_gar = data_qustop.loc[lab1 == 1]
data_nor = data_qustop.loc[lab1 == 0]
data_gar1 = cipin(data_gar, num=5)
data_nor1 = cipin(data_nor, num=30)

# 绘制垃圾短信词云图
# back_pic = imageio.imread('data/background.jpg')
back_pic = plt.imread('data/background.jpg')
wc = WordCloud(font_path='data/QingNiaoHuaGuangJianMeiHei-2.ttf',
               background_color='white',  # 背景颜色
               max_words=2000,  # 最大词数
               mask=back_pic,  # 背景图片
               max_font_size=200,  # 字体大小
               random_state=1234)  # 设置多少种随机的配色方案
gar_wordcloud = wc.fit_words(data_gar1)
plt.figure(figsize=(16, 8))
plt.imshow(gar_wordcloud)
plt.axis('off')
plt.savefig('output/spam_messages.jpg', dpi=600)
# plt.show()

# 绘制非垃圾短信词云图
nor_wordcloud = wc.fit_words(data_nor1)
plt.figure(figsize=(16, 8))
plt.imshow(nor_wordcloud)
plt.axis('off')
plt.savefig('output/non_spam_messages.jpg', dpi=600)
# plt.show()

# 代码8-5 数据采样
num = 10000
# 简单随机抽样，设置了随机种子，所以每次运行的结果都一样
adata = data_gar.sample(num, random_state=123)
bdata = data_nor.sample(num, random_state=123)
data_sample = pd.concat([adata, bdata])
cdata = data_sample.apply(lambda x: ' '.join(x))
lab = pd.DataFrame([1] * num + [0] * num, index=cdata.index)
my_data = pd.concat([cdata, lab], axis=1)
my_data.columns = ['message', 'label']

# 代码8-6 调用MultinomialNB函数进行分类和预测
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    my_data.message, my_data.label, test_size=0.2, random_state=123)  # 构建词频向量矩阵

# 训练集
cv = CountVectorizer()  # 将文本中的词语转化为词频矩阵
train_cv = cv.fit_transform(x_train)  # 拟合数据，再将数据转化为标准化格式
train_cv.toarray()
train_cv.shape  # 查看数据大小
cv.vocabulary_  # 查看词库内容

# 测试集
cv1 = CountVectorizer(vocabulary=cv.vocabulary_)
test_cv = cv1.fit_transform(x_test)
test_cv.shape
# 朴素贝叶斯
nb = MultinomialNB()  # 朴素贝叶斯分类器
nb.fit(train_cv, y_train)  # 训练分类器
pre = nb.predict(test_cv)  # 预测

# 代码8-7 模型评价
cm = confusion_matrix(y_test, pre)
cr = classification_report(y_test, pre)
print(cm)
print(cr)
