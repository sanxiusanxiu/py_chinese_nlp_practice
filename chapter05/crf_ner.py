# 代码5-4 基于CRF的命名实体预测
import pycrfsuite


# 定义一个函数，用于从单个词生成特征
def word2features(sent, i):
    word = sent[i][0]
    # 基础特征包括：词本身的小写、词尾的三个和两个字符、是否为数字、是否为空格
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isnumeric()': word.isnumeric(),
        'word.isspace()': word.isspace()
    }
    # 如果不是句子的第一个词，则添加前一个词的特征
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word[-3:]': prev_word[-3:],
            'prev_word[-2:]': prev_word[-2:],
            'prev_word.isnumeric()': prev_word.isnumeric(),
            'prev_word.isspace()': prev_word.isspace()
        })
    else:
        # 句子开始位置标记
        features['BOS'] = True
    # 如果不是句子的最后一个词，则添加下一个词的特征
    if i < len(sent) - 1:
        next_word = sent[i + 1][0]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word[-3:]': next_word[-3:],
            'next_word[-2:]': next_word[-2:],
            'next_word.isnumeric()': next_word.isnumeric(),
            'next_word.isspace()': next_word.isspace()
        })
    else:
        # 句子结束位置标记
        features['EOS'] = True
    return features


# 将句子转换为特征集的函数
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


# 从标注的句子中提取标签的函数
def sent2labels(sent):
    return [label for token, label in sent]


# 从标注的句子中提取单词的函数
def sent2tokens(sent):
    return [token for token, label in sent]


# 定义训练数据，每个元组包含单词及其标注
train_data = [
    [('我', 'O'), ('来自', 'O'), ('中国', 'LOC')],
    [('你', 'O'), ('是', 'O'), ('谁', 'PER')],
    [('他', 'O'), ('住', 'O'), ('在', 'O'), ('北京', 'LOC')]
]
# 生成训练数据的特征和标签
X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data]
# 初始化CRFsuite训练器，添加数据并设置参数
trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
trainer.set_params({
    'c1': 0.1,  # L1正则
    'c2': 0.01,  # L2正则
    'max_iterations': 100,  # 最大迭代次数
    'feature.possible_transitions': True  # 启用所有可能的转移特征
})
# 训练模型并保存
trainer.train('output/crf_model')

# 加载训练好的模型进行标注
tagger = pycrfsuite.Tagger()
tagger.open('output/crf_model')
test_data = [
    [('他', 'O'), ('在', 'O'), ('上海', 'LOC')],
    [('我', 'O'), ('在', 'O'), ('公司', 'ORG')]
]
# 生成测试数据的特征
X_test = [sent2features(s) for s in test_data]
y_pred = [tagger.tag(xseq) for xseq in X_test]
# 打印测试结果
for tokens, tags in zip(test_data, y_pred):
    for token, tag in zip(tokens, tags):
        print(token[0], tag)
    print('\n')
