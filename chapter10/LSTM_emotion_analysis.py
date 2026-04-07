# 代码10-7 读取语料数据
import pandas as pd

# 读取正负情感语料
neg = pd.read_excel('data/neg.xls', header=None, index_col=None)
pos = pd.read_excel('data/pos.xls', header=None, index_col=None)

# 给训练语料贴标签
pos['mark'] = 1
neg['mark'] = 0

# 代码10-8 词语向量化
from keras.preprocessing import sequence
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 分词
cut_word = lambda x: list(jieba.cut(x))  # 定义分词函数
pn_all = pd.concat([pos, neg], ignore_index=True)  # 合并正负情感语料
# 先确保所有输入都是字符串类型
pn_all[0] = pn_all[0].astype(str)

# 然后应用分词
cut_word = lambda x: list(jieba.cut(x))  # 定义分词函数
pn_all['words'] = pn_all[0].apply(cut_word)  # 对情感语料分词
comment = pd.read_excel('data/sum.xls')  # 读入评论内容,增加语料
comment = comment[comment['rateContent'].notnull()]  # 仅读取非空评论
comment['words'] = comment['rateContent'].apply(cut_word)  # 对评论语料分词
pn_comment = pd.concat([pn_all['words'], comment['words']], ignore_index=True)  # 合并所有的数据

# 正负情感评论词语向量化
w = []
for i in pn_comment:
    w.extend(i)
dicts = pd.DataFrame(pd.Series(w).value_counts())  # 建立统计词典
del w, pn_comment  # 删除临时文件 w，d2v_train
dicts['id'] = list(range(1, len(dicts) + 1))
get_sent = lambda x: list(dicts['id'][x])
pn_all['sent'] = pn_all['words'].apply(get_sent)

# 评论词语向量标准化，对样本进行padding填充和truncating修剪
max_len = 50  # 设置评论词语最大长度
pn_all['sent'] = list(sequence.pad_sequences(pn_all['sent'], maxlen=max_len))  # 正负情感评论词语向量化

# 训练集、测试集
x_all = np.array(list(pn_all['sent']))
y_all = np.array(list(pn_all['mark']))
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25)

print('训练集的特征数据形状为：', x_train.shape)
print('训练集的标签数据形状为：', y_train.shape)
print('测试集的特征数据形状为：', x_test.shape)
print('测试集的标签数据形状为：', y_test.shape)
print('训练集的特征数据为：\n', x_train)

# 代码10-9 模型构建
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras import Input

# 搭建LSTM模型
model = Sequential()
model.add(Input(shape=(50,)))
model.add(Embedding(len(dicts) + 1, 256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

# 代码10-10 模型训练
import time

# 设置超参
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
timeA = time.time()
model.fit(x_train, y_train, batch_size=16, epochs=10)
timeB = time.time()
print('time cost: ', int(timeB - timeA))

# 代码10-11 模型测试
from sklearn import metrics

# 使用测试数据进行模型测试
y_pred = model.predict(x_test).round().astype(int)
# 模型评价
acc = metrics.accuracy_score(y_test, y_pred)
print('测试集的准确率为：', acc)
print('精确率，召回率，F1值分别为：')
print(metrics.classification_report(y_test, y_pred))
print('混淆矩阵为：')
cm = metrics.confusion_matrix(y_test, y_pred)  # 混淆矩阵
print(cm)
