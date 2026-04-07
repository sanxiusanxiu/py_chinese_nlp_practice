

## 绪论



NLP是AI领域的重要分支，它使得计算机能够理解和生成人类语言

```python
# https://github.com/sanxiusanxiu/py_chinese_nlp_practice
```



#### NLP概述



###### NLP的研究内容



语言模型（预测下一个词出现的概率）



词法分析与句法分析



语义分析



语篇分析



机器翻译



情感分析



对话系统



信息检索、文本挖掘和知识图谱的构建



###### 学习NLP的挑战



字词组合复杂



语序灵活



多音字和同音字较多



语义表达隐含



语言资源较少



#### 中文NLP基本流程



语料获取，预料预处理，文本向量化，模型构建，模型训练，模型评价



#### 任务：构建中文文本高频词云图



注意放一个中文字体文件

![image-20260319175218716](D:\笔记\assets\image-20260319175218716.png)



## 语料库



#### 语料库概述



语料库是为了支持和促进NLP技术的研究与开发，收集和组织的大量文本或语音数据集合



#### 语料库的获取



搜狗新闻语料库，来源：https://gitcode.com/Premium-Resources/f08cf

![image-20260405154252238](D:\笔记\assets\image-20260405154252238.png)



中文人民日报语料库，来源：https://github.com/chenhui-bupt/PeopleDaily1998



中文社交媒体文本，来源：https://wenku.csdn.net/doc/84s6qs1fst，https://gitcode.com/open-source-toolkit/56cad



中文电子文本项目，来源：https://ctext.org/instructions/library/zh



百度中文问答数据集，来源：https://blog.csdn.net/weixin_48827824/article/details/129243125，https://github.com/brightmart/nlp_chinese_corpus，https://github.com/SophonPlus/ChineseNlpCorpus/



清华大学开放中文语料库，来源：https://blog.gitcode.com/da75a86c17f73f8072f14b31e7fe7cab.html



补充：https://blog.csdn.net/OpenBayes/article/details/135859340



#### 任务：网络在线语料分析



在线古腾堡语料库是一个涵盖多文学领域的中文图书电子书库，可以从这里下载目录，寻找对应的图书id：https://www.gutenberg.org/cache/epub/feeds/

![image-20260319212610074](D:\笔记\assets\image-20260319212610074.png)



![image-20260319213541839](D:\笔记\assets\image-20260319213541839.png)



#### 任务：构建一个电影评论语料库



使用ratings.csv构建小型电影评论语料库

![image-20260320164353749](D:\笔记\assets\image-20260320164353749.png)



#### 任务：创建武侠小说语料库



使用Trae的SOLO模式，根据本章节的三段代码完成本次任务

![image-20260405173405207](D:\笔记\assets\image-20260405173405207.png)



#### 任务：创建伤寒论语料库



不过效果并不怎么好

![image-20260405182920302](D:\笔记\assets\image-20260405182920302.png)



## 正则表达式



这部分可以和爬虫的笔记进行合并



#### 正则表达式的应用



数据验证与格式化，文本搜索与信息抽取，数据清洗与转换，日志文件分析，语法分析与文本解析



#### 正则表达式函数



###### match



###### search



###### findall



###### sub



###### finditer



###### split



#### 正则表达式元字符



###### 量词元字符



###### 字符类元字符



###### 锚点和边界元字符



###### 特殊元字符



#### 任务：正则的简单应用



```python
import re

# 代码3-17 验证电子邮件地址格式
def match_email_address(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if re.match(pattern, email):
        return True
    else:
        return False
email1 = "test@example.com"
email2 = "invalid_email"
print(match_email_address(email1))  # 输出 True
print(match_email_address(email2))  # 输出 False

# 代码3-18 验证电话号码
def validate_phone_number(phone_number):
    pattern = r'^\d{3}-\d{8}$'
    if re.match(pattern, phone_number):
        return True
    else:
        return False
phone_number = '123-45678901'
if validate_phone_number(phone_number):
    print("电话号码格式正确")
else:
    print("电话号码格式不正确")

# 代码3-19 验证日期格式
def validate_date(date):
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if re.match(pattern, date):
        return True
    else:
        return False
date = '2023-10-27'
if validate_date(date):
    print("日期格式正确")
else:
    print("日期格式不正确")

# 代码3-20 提取文本中姓名、性别、年龄、电话号码和地址等信息
text = """
姓名：张三
性别：男
年龄：25岁
电话号码：13512345678
地址：北京市朝阳区
"""
name_pattern = r"姓名：(.*?)\n"
gender_pattern = r"性别：(.*?)\n"
age_pattern = r"年龄：(.*?)岁\n"
phone_pattern = r"电话号码：(.*?)\n"
address_pattern = r"地址：(.*?)\n"
name = re.search(name_pattern, text).group(1)
gender = re.search(gender_pattern, text).group(1)
age = re.search(age_pattern, text).group(1)
phone = re.search(phone_pattern, text).group(1)
address = re.search(address_pattern, text).group(1)
print("姓名:", name)
print("性别:", gender)
print("年龄:", age)
print("电话号码:", phone)
print("地址:", address)

```



## 中文分词



#### 中文分词简介



中文分词是指将连续的中文文本切割成具有独立语义的词语的过程



###### 中文分词的难点



词边界不明确，歧义词的处理，新词的识别，专有名词和缩写词，语境和语义依赖性



###### 中文分词的方法



基于规则的分词



基于统计的分词



基于深度学习的分词



#### 基于规则的分词



基于规则的分词利用预定义词典和分词规则对中文文本进行切分，主要通过匹配词典中的词条和应用分词规则来识别词语边界，此外，还需处理词语歧义和识别未登录词



基于规则的分词简单直接，具有较强的可控性，但依赖于完备的词典，分词规则维护复杂，且在处理复杂上下文和歧义问题时能力有限



###### 正向最大匹配法



![image-20260322154229345](D:\笔记\assets\image-20260322154229345.png)



###### 逆向最大匹配法



![image-20260322154330653](D:\笔记\assets\image-20260322154330653.png)



###### 双向最大匹配法



![image-20260322154543157](D:\笔记\assets\image-20260322154543157.png)



#### 基于统计的分词



基于统计的分词利用统计学原理来解决中文文本的分词问题，核心在于通过分析大量语料数据，学习词语之间的统计规律，进而自动识别出文本中词的边界



基于统计的分词主要依赖于概率模型和语料库



###### 马尔科夫模型



未来状态仅依赖于当前状态，与历史状态无关，通俗来说就是“下一个字是什么，只跟当前这个字有关，跟更早的字无关。”



马尔科夫模型的思路是给每个字打标签，标签只有四种：B，词的开始 (Begin)；M，词的中间 (Middle)；E，词的结尾 (End)；S，单字成词 (Single)



例如，“南京市长江大桥”，得到：南(B) 京(M) 市(E) 长(B) 江(E) 大(B) 桥(E)



###### 隐马尔科夫模型



HMM隐马尔科夫模型，适合时间序列数据



HMM有两个关键概率：发射概率，某个标签下出现某个字的概率，比如标签是B时，出现"南"的概率；转移概率，从一个标签跳到另一个标签的概率，比如B后面跟着M的概率；最后再加上一个初始概率（第一个字是B/E/S的概率）

![image-20260405223224101](D:\笔记\assets\image-20260405223224101.png)



#### 基于深度学习的分词



基于深度学习的中文分词利用神经网络自动学习语言规律，无需人工规则，能更好地处理歧义、上下文和新词识别，显著提升分词效果



基于深度学习的分词通过学习大量标注好的数据，能够自动地从输入文本中学习分词的规律和模式



主要包括数据预处理、特征提取、模型选择、模型训练、分词预测五个步骤



###### 中文分词实例



```python
# 深度学习分词方法主要步骤
#
# 整体思路：把分词问题转化为"序列标注"问题
# 即：给句子中每个字打标签（B/I/S），然后根据标签切词
#   B = 词的开头（Begin）
#   I = 词的中间（Inside）
#   S = 单字成词（Single）
#
# 举例："我爱中国" → 我(S) 爱(S) 中(B) 国(I) → ["我", "爱", "中国"]
#
# 模型选择：BiLSTM（双向长短期记忆网络）
#   - 双向：同时看左边和右边的上下文，解决"只看前一个字"的局限
#   - 相比 HMM：不需要手工统计转移概率/发射概率，模型自己学

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ==================== 数据准备 ====================
# 训练数据格式：(原始句子, 正确的分词结果)
# 数据量很小，仅用于演示流程；实际应用需要数万到数十万条
data = [
    ("我爱中国", ["我", "爱", "中国"]),
    ("我爱北京天安门", ["我", "爱", "北京", "天安门"]),
    ("中华人民共和国万岁", ["中华人民共和国", "万岁"]),
    ("今天天气不错", ["今天", "天气", "不错"]),
    ("我们去公园玩", ["我们", "去", "公园", "玩"]),
    ("他昨天去了图书馆", ["他", "昨天", "去", "了", "图书馆"]),
    ("中国的发展速度很快", ["中国", "的", "发展", "速度", "很快"]),
    ("请你调低一点音量", ["请", "你", "调低", "一点", "音量"]),
    ("学生会主席发表了讲话", ["学生会", "主席", "发表", "了", "讲话"]),
    ("她的笑容很迷人", ["她", "的", "笑容", "很", "迷人"]),
    ("大家一起来帮忙", ["大家", "一起", "来", "帮忙"])
]


# ==================== 构建词汇表和标签表 ====================
# 词汇表：把每个字映射成一个数字 ID，神经网络只认数字
# <PAD> 是填充符（让所有句子等长），<UNK> 是未知字（词表里没有的字）
def build_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sentence, _ in data:
        for char in sentence:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


# 标签表：只有 4 个标签，对应 BIO 标注方案
def build_tag_vocab():
    # O=非词(填充用), B=词首, I=词中, S=单字词
    return {"O": 0, "B": 1, "I": 2, "S": 3}  


vocab = build_vocab(data)
tag_vocab = build_tag_vocab()


# ==================== 数据预处理 ====================
# 核心步骤：把"分词结果"转换成"每个字的 BIO 标签"
# 这样模型就知道：每个字应该被标什么标签
def generate_bio_tags(sentence, words):
    bio_tags = []
    index = 0
    for word in words:
        for i in range(len(word)):
            if len(word) == 1:
                # 单字词 → 标 S
                bio_tags.append("S")
            elif i == 0:
                # 多字词的第一个字 → 标 B
                bio_tags.append("B")
            else:
                # 多字词的后续字 → 标 I
                bio_tags.append("I")
        index += len(word)
    return bio_tags


# 把原始数据转成模型能吃的数字格式
def preprocess_data(data, vocab, tag_vocab):
    inputs, labels = [], []
    for sentence, words in data:
        # 每个字 → 对应的数字 ID（词表里没有的字用 <UNK> 的 ID）
        input_ids = [vocab.get(char, vocab["<UNK>"]) for char in sentence]
        # 生成本句的 BIO 标签序列
        bio_tags = generate_bio_tags(sentence, words)
        # 标签也转成数字
        label_ids = [tag_vocab[tag] for tag in bio_tags]
        inputs.append(input_ids)
        labels.append(label_ids)
    return inputs, labels


inputs, labels = preprocess_data(data, vocab, tag_vocab)


# ==================== 动态填充 ====================
# 神经网络要求每个 batch 的输入形状一致（矩阵），所以短句子要补齐
# 比如 ["我", "爱", "中国"] 长度 3，要补到和最长句子一样长
def pad_sequences(sequences, pad_value):
    max_length = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]


# 输入用 <PAD>(0) 填充，标签用 O(0) 填充（O 表示"非词"，计算 loss 时可忽略）
inputs = pad_sequences(inputs, vocab["<PAD>"])
labels = pad_sequences(labels, tag_vocab["O"])

# ==================== 数据划分 ====================
# 80% 训练，20% 测试（实际项目中还需要验证集）
split_idx = int(0.8 * len(inputs))
train_data = list(zip(inputs[:split_idx], labels[:split_idx]))
test_data = list(zip(inputs[split_idx:], labels[split_idx:]))


# ==================== 定义模型：BiLSTM ====================
# 结构：输入 → Embedding → BiLSTM → 全连接层 → 输出(每个位置4个标签的分数)
#
#        "我" "爱" "中" "国"
#         ↓    ↓    ↓    ↓       ← 输入：字的 ID
#       [Embedding 层：把每个字变成一个稠密向量]
#         ↓    ↓    ↓    ↓
#       [BiLSTM 层：双向扫描，捕获上下文信息]
#         ↓    ↓    ↓    ↓
#       [全连接层：输出每个位置属于 B/I/S/O 的分数]
#         ↓    ↓    ↓    ↓
#        S    S    B    I        ← 输出：预测的标签
#
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Embedding 层：把离散的字 ID 映射成连续的向量
        # 类似 word2vec 的思想，让语义相近的字在向量空间中靠近
        # input_size=词表大小, hidden_size=每个字的向量维度
        self.embedding = nn.Embedding(input_size, hidden_size)

        # BiLSTM 层：双向 LSTM
        # hidden_size // 2 是因为双向拼接后总维度 = hidden_size
        # 正向 LSTM 看左边的上下文，反向 LSTM 看右边的上下文
        # 拼在一起就能同时利用左右两侧的信息
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, bidirectional=True)

        # 全连接层：把 LSTM 的输出映射到标签空间
        # 输入维度 = hidden_size（双向拼接后），输出维度 = 标签数(4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # 1. 字 ID → 字向量 (batch_size, seq_len, hidden_size)
        embedded = self.embedding(inputs)

        # 2. 字向量 → 上下文表示 (batch_size, seq_len, hidden_size)
        # outputs 是每个时间步的隐藏状态，_ 是最终的 (h_n, c_n)，这里不需要
        outputs, _ = self.lstm(embedded)

        # 3. 上下文表示 → 标签分数 (batch_size, seq_len, output_size)
        # 每个位置输出 4 个分数，分别对应 O/B/I/S
        logits = self.fc(outputs)
        return logits


# ==================== 模型初始化 ====================
input_size = len(vocab)        # 词表大小（字的数量 + 特殊符号）
hidden_size = 128              # 隐藏层维度，太小学不好，太大容易过拟合
output_size = len(tag_vocab)   # 标签数量 = 4 (O, B, I, S)

model = BiLSTM(input_size, hidden_size, output_size)

# 损失函数：交叉熵，分类任务的标准选择
# 本质上就是衡量"模型预测的标签分布"和"真实标签"之间的差距
criterion = nn.CrossEntropyLoss()

# 优化器：Adam，自适应学习率，比 SGD 更省心
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 16


# ==================== 训练函数 ====================
def train_model(train_data, model, criterion, optimizer, batch_size):
    # DataLoader 自动把数据分成小批次（mini-batch）
    # collate_fn=lambda x: x 保持原始格式，因为我们在循环里手动填充
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=lambda x: x)

    for epoch in range(10):  # 训练 10 轮（实际项目可能需要几十到上百轮）
        train_loss = 0
        for batch in train_loader:
            # 解包当前批次的数据
            inputs, labels = zip(*batch)

            # 把列表转成 tensor，并填充到相同长度
            inputs_tensor = torch.tensor(pad_sequences(inputs, vocab["<PAD>"]))
            labels_tensor = torch.tensor(pad_sequences(labels, tag_vocab["O"]))

            # --- 标准训练流程：前向 → 算损失 → 反向传播 → 更新参数 ---
            # 1. 清空上一步的梯度
            optimizer.zero_grad()
            # 2. 前向传播：输入 → 模型 → 预测
            outputs = model(inputs_tensor)
            # 3. 计算损失
            # view(-1, output_size) 把 (batch, seq_len, 4) 展平成 (batch*seq_len, 4)
            # view(-1) 把标签也展平成一维，一一对应计算每个位置的误差
            loss = criterion(outputs.view(-1, output_size), labels_tensor.view(-1))
            # 4. 反向传播：计算梯度
            loss.backward()
            # 5. 更新参数：沿梯度方向调整权重
            optimizer.step()

            train_loss += loss.item()
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss / len(train_data)}")


train_model(train_data, model, criterion, optimizer, batch_size)


# ==================== 预测函数 ====================
# 给一句话，输出每个字的 BIO 标签
def predict_sentence(sentence, vocab, model):
    # 把句子转成数字 ID
    input_ids = [vocab.get(char, vocab["<UNK>"]) for char in sentence]
    # 包装成 tensor，形状 (1, seq_len)——batch_size=1
    inputs_tensor = torch.tensor([input_ids])

    # 模型推理（不需要计算梯度）
    with torch.no_grad():
        outputs = model(inputs_tensor)

    # 取每个位置分数最高的标签 ID
    # dim=2 表示在标签维度上取最大值
    _, predicted = torch.max(outputs.data, 2)

    # 把数字 ID 转回标签字符串
    predicted_tags = [list(tag_vocab.keys())[tag_id] for tag_id in predicted[0]]
    return predicted_tags


# 把 BIO 标签序列还原成分词结果
# 核心逻辑：B 开始一个新词，I 继续当前词，S 直接成一个词
def tags_to_words(sentence, tags):
    words = []
    word = ''
    for char, tag in zip(sentence, tags):
        if tag == 'B':
            # 遇到 B：之前的词结束（如果有），开始新词
            if word:
                words.append(word)
            word = char
        elif tag == 'I':
            # 遇到 I：继续往当前词里追加字
            word += char
        elif tag == 'S':
            # 遇到 S：之前的词结束（如果有），当前字单独成词
            if word:
                words.append(word)
                word = ''
            words.append(char)
    # 别忘了最后一个词
    if word:
        words.append(word)
    return words


# ==================== 测试预测功能 ====================
test_sentence = "我爱中国"
predicted_tags = predict_sentence(test_sentence, vocab, model)
print("Predicted BIO Tags:", predicted_tags)

# 将 BIO 标签转换为词
words = tags_to_words(test_sentence, predicted_tags)
print("Segmented Words:", words)

```



![image-20260322162446654](D:\笔记\assets\image-20260322162446654.png)



#### 中文分词jieba



jieba是一个开源中文分词工具，其提供基于前缀词典和HMM的中文分词功能



###### jieba算法简介



基于前缀词典、HMM、全局剪枝策略、基于用户自定义的词典



###### jieba分词模式



精确模式（默认），具有较高的准确性



全模式，会生成较多的分词结果，但准确性相对较低，适用于词频统计、快速查询等对分词结果不太严格的场景



搜索引擎模式，基于精确模式，适用于对分词准确性和效果要求比较高的场景



![image-20260322163054834](D:\笔记\assets\image-20260322163054834.png)



#### 任务：中文分词的应用



###### HMM中文分词



HMM中文分词的基本步骤如下：构建分词语料库，确定隐含状态和观测状态，构建初始状态概率分布，构建状态转移概率矩阵，构建发射概率矩阵，构建HMM，分词预测

```python
# 代码4-12 训练HMM模型
# 若要二次运行，则需删除已生成的json文件，否则会继续对原文件写入内容并出现解析错误
import os
import json
import datetime

text = '学校是学习的好地方！'

def train():
    # 初始化参数
    trans_prob = {}  # 转移概率
    emit_prob = {}  # 发射概率
    init_prob = {}  # 状态出现次数
    state_list = ['B', 'M', 'E', 'S']
    Count_dict = {}
    for state in state_list:
        trans = {}
        for s in state_list:
            trans[s] = 0
        trans_prob[state] = trans
        emit_prob[state] = {}
        init_prob[state] = 0
        Count_dict[state] = 0
    count = -1
    # 读取并处理单词、计算概率矩阵
    path = 'input/trainCorpus.txt'
    for line in open(path, 'r', encoding='gbk'):
        count += 1
        line = line.strip()
        if not line:
            continue
        # 读取每一行的单词
        word_list = []
        for i in line:
            if i != ' ':
                word_list.append(i)
        # 标注每个单词的位置标签
        word_label = []
        for word in line.split():
            label = []
            if len(word) == 1:
                label.append('S')
            else:
                label += ['B'] + ['M'] * (len(word) - 2) + ['E']
            word_label.extend(label)
        # 统计各个位置状态下的出现次数，用于计算概率
        for index, value in enumerate(word_label):
            Count_dict[value] += 1
            if index == 0:
                init_prob[value] += 1
            else:
                trans_prob[word_label[index - 1]][value] += 1
                emit_prob[word_label[index]][word_list[index]] = (
                        emit_prob[word_label[index]].get(
                            word_list[index], 0) + 1.0)
    # 初始概率
    for key, value in init_prob.items():
        init_prob[key] = value * 1 / count
    # 转移概率
    for key, value in trans_prob.items():
        for k, v in value.items():
            value[k] = v / Count_dict[key]
        trans_prob[key] = value
    # 发射概率，采用加1平滑
    for key, value in emit_prob.items():
        for k, v in value.items():
            value[k] = (v + 1) / Count_dict[key]
        emit_prob[key] = value
    # 将3个概率矩阵保存至json文件
    model = 'output/hmm_model.json'
    with open(model, 'w', encoding='utf-8') as f:
        f.write(json.dumps(init_prob) + '\n')
        f.write(json.dumps(trans_prob) + '\n')
        f.write(json.dumps(emit_prob) + '\n')


# 代码4-13 实现viterbi算法
def viterbi(text, state_list, init_prob, trans_prob, emit_prob):
    V = [{}]
    path = {}
    # 初始概率
    for state in state_list:
        V[0][state] = init_prob[state] * emit_prob[state].get(text[0], 0)
        path[state] = [state]
    # 当前语料中所有的字
    key_list = []
    for key, value in emit_prob.items():
        for k, v in value.items():
            key_list.append(k)
    # 计算待分词文本的状态概率值，得到最大概率序列
    for t in range(1, len(text)):
        V.append({})
        newpath = {}
        for state in state_list:
            if text[t] in key_list:
                emit_count = emit_prob[state].get(text[t], 0)
            else:
                emit_count = 1
            (prob, a) = max(
                [(V[t - 1][s] * trans_prob[s].get(state, 0) * emit_count, s)
                 for s in state_list if V[t - 1][s] > 0])
            V[t][state] = prob
            newpath[state] = path[a] + [state]
        path = newpath
    # 根据末尾字的状态，判断最大概率状态序列
    if emit_prob['M'].get(text[-1], 0) > emit_prob['S'].get(text[-1], 0):
        (prob, a) = max([(V[len(text) - 1][s], s) for s in ('E', 'M')])
    else:
        (prob, a) = max([(V[len(text) - 1][s], s) for s in state_list])
    return (prob, path[a])


# 代码4-14 HMM中文分词
def cut(text):
    state_list = ['B', 'M', 'E', 'S']
    model = 'output/hmm_model.json'
    # 先检查当前路径下是否有json文件，如果有json文件，需要删除
    if os.path.exists(model):
        with open(model, 'r', encoding='utf-8') as f:
            init_prob = json.loads(f.readline())
            trans_prob = json.loads(f.readline())
            emit_prob = json.loads(f.readline())
    else:
        trans_prob = {}
        emit_prob = {}
        init_prob = {}
    # 利用维特比算法，求解最大概率状态序列
    prob, pos_list = viterbi(text, state_list, init_prob, trans_prob, emit_prob)
    # 判断待分词文本每个字的状态，输出结果
    begin, follow = 0, 0
    for index, char in enumerate(text):
        state = pos_list[index]
        if state == 'B':
            begin = index
        elif state == 'E':
            yield text[begin: index + 1]
            follow = index + 1
        elif state == 'S':
            yield char
            follow = index + 1
    if follow < len(text):
        yield text[follow:]


# 训练、分词
start_time = datetime.datetime.now()
train()
end_time = datetime.datetime.now()
print("训练用时：", (end_time - start_time))
cut(text)
print(text)
print(str(list(cut(text))))

```



![image-20260322164918351](D:\笔记\assets\image-20260322164918351.png)



###### 提取新闻文本的高频词



提取新闻文本的高频词是一种常用的文本分析方法，它可以帮助了解新闻文本的关键信息和热点话题

![image-20260322165226058](D:\笔记\assets\image-20260322165226058.png)



提取新闻文本高频词2

```python
import requests
from bs4 import BeautifulSoup
import jieba
from opencc import OpenCC
from collections import Counter
import re

# 读取新闻文本
with open("input/flightnews.txt", "r", encoding="utf-8") as f:
    text = f.read()
print("文本长度:", len(text))

# 加载停用词
stop_words = set()
with open('input/stopword.txt', 'r', encoding='utf-8') as f:
    for word in f:
        stop_words.add(word.strip())

# 清洗文本，移除非中文字符
text_cleaned = re.sub(r'[^\u4e00-\u9fa5。,，！!?？]', '', text)
# 使用jieba进行分词
words = jieba.cut(text_cleaned)
# Python 3.8+ 引入的海象运算符 :=，在判断非空的同时把结果赋值给了 cleaned 变量
filtered_words = [
    word for word in words
    if (cleaned := word.strip()) and cleaned not in stop_words
]
# 方式一
# filtered_words = [word for word in words if word.strip() and word.strip() not in stop_words]
# 方式二
# filtered_words = []
# for word in words:
#     cleaned = word.strip()
#     if cleaned and cleaned not in stop_words:
#         filtered_words.append(word)

# 统计词频
word_counts = Counter(filtered_words)
# 输出基础统计数据
print("文本总词数：", len(filtered_words))
print("不重复词汇总数：", len(word_counts))
# 输出高频词统计
top_n = 10
print(f"出现频率最高的{top_n}个词：")
for word, count in word_counts.most_common(top_n):
    print(word, ":", count)

```



## 词性标注和命名实体识别



#### 词性标注简介



词性标注就是为文本中的每个词确定词性的过程



中文词性标注通常基于大规模已标注好词性的中文语料库进行训练，采用统计模型、机器学习模型或深度学习模型推断每个词可能的词性



常见的中文词性标注包括名词（N）、动词（V）、形容词（ADJ）、副词（ADV）、代词（PRON）、数词（NUM）、量词（M）、冠词（DET）、介词（PREP）、连词（CONJ）、助动词（AUX）等



#### 词性标注模型



###### HMM



HMM用于词性标注时，将每个词语视为观测状态，而其词性则被视为隐藏状态，HMM通过观测序列（即词语序列）来推断隐藏状态序列（即词性标签序列）

![image-20260323101614289](D:\笔记\assets\image-20260323101614289.png)



###### 最大熵模型



熵在物理学和信息论中代表混乱度或不确定性，即在满足已知约束条件的前提下，让未知部分的不确定性最大化（概率分布尽可能均匀）



###### 深度学习模型



通过使用深度学习模型来进行词性标注，能够更好地捕捉上下文信息和语义特征，进而提升词性标注的准确性

![image-20260323101915762](D:\笔记\assets\image-20260323101915762.png)



#### 基于jieba的词性标注



jieba的词性标注采用了ICTCLAS词性标注集，该标注集包括很多常见的词性类别，例如名词、动词、形容词、副词等

![image-20260406083519396](D:\笔记\assets\image-20260406083519396.png)



#### 命名实体识别



命名实体识别（Named Entity Recognition，NER）是一种从文本中识别出具有特定意义的命名实体的技术，例如识别一段文本中的人名、地名、组织机构名等，通常是指代表着具体实体或具有特定指向的词语或短语



###### 基于CRF模型



CRF是一种广泛用于标注和分割序列数据的统计建模方法



CRF把感知机的特征能力，以及HMM的序列建模能力结合在一起，全局地考虑整个序列的标签组合，选出概率最大的那组

![image-20260323105312179](D:\笔记\assets\image-20260323105312179.png)



###### 基于深度学习



利用深度学习模型自动识别中文文本中的命名实体

![image-20260323105418166](D:\笔记\assets\image-20260323105418166.png)



#### 任务：使用sklearn-crfsuite进行中文NER



使用sklearn-crfsuite可以通过提供输入序列的特征和对应的标签序列，训练一个CRF模型，也可以使用训练好的模型对新的序列进行预测，并获得相应的标注结果

![image-20260323113749302](D:\笔记\assets\image-20260323113749302.png)



## 关键词提取



#### 关键词提取简介



关键词提取是从单个文本或一个语料库中，根据核心词语的统计和语义分析，选择适当的、能够完整表达主题内容的词语的过程



#### 基于统计的方法



基于统计的方法主要通过分析文本中的统计特征来识别和提取关键词



这类方法的核心思想是，文本中的关键信息往往可以通过词的使用频率和分布模式等统计属性来反映



###### TF-IDF算法



TF-IDF算法的主要思想是字词的重要性随着它在文档中出现次数的增加而上升，并随着它在语料库中出现频率的升高而下降



词频（Term Frequency）， $TF(word,doc) = \frac{词word在文档doc中出现的次数}{文档doc的总词数}$ 



逆文档频率（Inverse Document Frequency），$IDF(word) = log\frac{总文档数}{包含词word的文档数 + 1}$ 



TF-IDF(词w, 文档d) = TF(词w, 文档d) × IDF(词w)



假设文档集共有2000篇文档，包含“孩子们”，“快乐”，“都是”，“他们”，“大山”这几个词的文档数分别为60、30、250、200、20，每个词的TF值都为0.033，IDF值分别为1.516、1.810、0.901、0.998、1.979



根据TF-IDF算法的计算公式，将每个词语的TF值和IDF值相乘，得到5个词语的TF-IDF值分别为0.0500、0.0597、0.0297、0.0329、0.0653，因此，选取TF-IDF值相对较大的前3个关键词，即“大山”“快乐”“孩子们”作为这篇文档的关键词

![image-20260323174443743](D:\笔记\assets\image-20260323174443743.png)



###### TextRank算法



TextRank算法基于PageRank算法的思想，将文本中的句子或词表示为图中的节点，通过节点之间的边表示它们之间的关联关系，然后利用图的计算方法来确定每个节点的重要性，从而得到关键句或关键词



补充：PageRank的核心思想，一个网页如果被很多重要网页链接，那它本身也很重要



TextRank算法通俗来说就是，一个词如果和很多重要词共现，那它本身也很重要



TextRank算法提取中文关键词的步骤可以分为分词（如jieba）、词性标注和过滤（如去除停用词）、构建图（以词为节点构建图）、权重计算、关键词选取，迭代到分数收敛时（变化小于某个阈值），排序取Top N就是关键词



构建图时，常常设定一个窗口大小（比如5个词），窗口内共现的词（重复出现的词）之间连一条边



迭代公式，$WS(V_i) = (1 - d) + d \sum\limits_{V_j \in In(V_i)} (\frac{w_{ji}}{\sum\limits_{V_k \in Out(V_j)} w_{jk}} \times WS(V_j))$ 

$WS(V_i)$ 是 $V_i$ 的TextRank分数

$d$ 是阻尼系数，通常取0.85（和PageRank一样）

$In(V_i)$ 是指向 $V_i$ 的所有邻居词（有边相连的词）

$w_{ji}$ 是词 $V_j$ 到词 $V_i$ 的边权重（共现次数越多，权重越大）

$Out(V_j)$ 是词 $V_j$ 的所有邻居词

$WS(V_j)$ 是邻居词 $V_j$ 的当前分数

![image-20260323180623626](D:\笔记\assets\image-20260323180623626.png)



#### 基于语义的方法



基于语义的方法是指通过考虑词语之间的语义关系来进行关键词提取



这类方法尝试从文本中抽取语义相对丰富且具有代表性的词语作为关键词，而不仅仅依赖于词频或其他统计信息



###### LSA算法



LSA（Latent Semantic Analysis，潜在语义分析）通过数学方法挖掘文本数据中的潜在语义关系，假设词的使用模式反映深层语义结构，通过奇异值分解（SVD）分析词-文档矩阵来发现词语和文档之间的模式



LSA算法提取中文关键词的步骤可以分为构建词-文档矩阵、奇异值分解、降维、提取关键词

![image-20260324113444235](D:\笔记\assets\image-20260324113444235.png)



###### LDA算法



LDA（Latent Dirichlet Allocation，潜在狄利克雷分配）假定词语之间没有顺序，所有的词语都无序地放在一个袋子里，并且认为一个文档可以有多个主题，每个主题对应不同的词语



LSA是"矩阵分解"，LDA是"概率生成"

![image-20260324114052155](D:\笔记\assets\image-20260324114052155.png)



#### 任务：自动提取文本关键词



![image-20260324120205607](D:\笔记\assets\image-20260324120205607.png)



## 文本向量化



#### 文本向量化简介



将文本数据转换为数值数据的过程，以便计算机能够有效地处理和分析文本



#### 离散化表示



文本向量化的离散化表示是将文本数据转换为一系列离散数值的过程，以便计算机能够进行处理和分析



这种表示方式通常涉及将文本中的每个词或短语映射到一个高维空间中的向量，其中每个维度代表词汇表中的一个特定词，而每个维度上的值通常是一个表示该词在文档中是否出现（独热编码）或出现频率（词袋模型）的离散数值



###### 独热编码



是一种将文本数据转换为数值向量的方法，特别适合将词、字符或其他文本单元转换成数值格式，以便计算机能够处理

![image-20260327101315480](D:\笔记\assets\image-20260327101315480.png)



###### BOW模型



BOW模型是一种简单有效的文本向量化方法，它将文本转换为固定长度的向量



但BOW模型无法捕捉词语之间的顺序信息，适用于一些不依赖于词语顺序的任务，如情感分类、文本聚类等

![image-20260327101503412](D:\笔记\assets\image-20260327101503412.png)



###### TF-IDF方法



TF-IDF方法是在BOW模型的基础上进一步优化的一种方法，它通过考虑词语在语料库中的IDF来调整词频，以此减少常见词的影响并突出重要词语

![image-20260327101544391](D:\笔记\assets\image-20260327101544391.png)



#### 分布式表示



文本向量化的分布式表示是指通过将文本中的词语映射到一个连续的向量空间中，以便捕获和表示词语的语义信息



与离散化表示不同，分布式表示能够反映词语之间的相似性，为词语在具体语境中的运用提供了丰富的语义信息



###### 词嵌入



词嵌入是文本向量化的分布式表示中最常见的方法，它将每个词表示为一个固定长度的稠密向量



这些向量通常通过训练基于大型文本语料库的神经网络模型获得，如Word2Vec、GloVe和fastText等



Word2Vec通过两种模型（CBoW模型和Skip-Gram模型）训练词向量，捕获词语间的复杂语义关系和语法规律



GloVe结合了全局矩阵分解和局部上下文窗口的优点，通过共现矩阵来训练词向量



fastText类似于Word2Vec，但它将词拆分为n元语法（n-grams），从而更好的处理罕见词和新词



###### 句子和文档嵌入



除了单个词的向量表示，分布式表示方法也被用于生成整个句子或文档的向量表示



文档嵌入Doc2Vec是Word2Vec的扩展，能够学习到文档级别的向量表示



句子嵌入（如Sentence-BERT）利用预训练的深度学习模型（如BERT），通过特定的策略（如平均词向量、最大化注意力权重等）来获得整个句子的向量表示



###### Word2Vec模型



Word2Vec模型将词语转换为一组表示语义信息的向量，能够捕捉到词语之间的复杂关系，如语义相似性、词的上下文关联等，在执行NLP任务时能更好地理解文本内容



Word2Vec模型基于这样一个假设：在相似的上下文中出现的词语往往具有相似的语义



因此，Word2Vec模型通过学习词语的上下文来推断词语的语义，使语义相近的词语在向量空间中彼此接近



使用CBoW模型实现中文文本向量化，CBoW模型的主要目标是利用给定的目标词的上下文来预测目标词本身

![image-20260327101718213](D:\笔记\assets\image-20260327101718213.png)



使用Skip-Gram模型实现中文文本向量化，其核心任务是根据当前的目标词预测其上下文中的词语，因而非常擅长捕获词之间的远距离依赖关系

![image-20260327101854905](D:\笔记\assets\image-20260327101854905.png)



###### Doc2Vec模型



Doc2Vec模型是Word2Vec模型的扩展，不仅能够为词生成向量表示，还能够为更长的文本序列生成向量表示



Doc2Vec模型特别适合用于文本相似性度量、文档聚类、文本分类等需要理解和比较整个文本的含义的任务



使用DM模型实现中文文本向量化，DM模型的核心思想是，不仅词语可以通过其上下文被有效地表示，整个文档或段落也可以通过其包含的词语的上下文来有效地表示

![image-20260327102231320](D:\笔记\assets\image-20260327102231320.png)



DBoW模型是Doc2Vec模型的一种变体，用于学习文本的向量表示，DBoW模型与DM模型并行存在，两者共同构成了Doc2Vec模型的基础



DBoW模型采用了一种更为直接的方法，忽略上下文中的词序，直接使用文档的ID，即文档的向量表示预测文档中的词

![image-20260327102335985](D:\笔记\assets\image-20260327102335985.png)



#### 任务：文本相似度计算



NLP任务离不开语料数据的支撑，词向量的训练也不例外，词向量的训练可以分为两个步骤进行，先对中文语料进行预处理，然后使用gensim库训练词向量



###### 因设备原因，暂时使用教材配套给出的模型



reduce_zhiwiki文件下载链接：https://pan.baidu.com/s/1DdL2Fen-kMobYmnDE4KSuA 



###### Word2Vec词向量的训练



![image-20260406170733214](D:\笔记\assets\image-20260406170733214.png)



###### Doc2Vec段落向量的训练



![image-20260406171809523](D:\笔记\assets\image-20260406171809523.png)



###### 计算文本相似度



使用Word2Vec模型计算文本相似度，首先提取关键词，然后得到文本向量化，最后计算文本相似度

![image-20260406172348040](D:\笔记\assets\image-20260406172348040.png)



使用Doc2Vec模型计算文本相似度，首先数据预处理，然后文本向量化，最后计算相似度

![image-20260406172529914](D:\笔记\assets\image-20260406172529914.png)



## 文本分类和文本聚类



#### 文本挖掘简介



文本挖掘是指从大量文本数据中自动地发现并提取有意义的信息和知识的过程



文本挖掘的主要包括文本预处理、特征提取、模型构建和评估等



#### 文本分类



文本分类是将一段文本自动分配到一个或多个预先定义的类别中



文本分类算法可以大致分为基于传统机器学习的文本分类算法和基于深度学习的文本分类算法



###### 基于传统机器学习



朴素贝叶斯，SVM，决策树，随机森林，逻辑回归，KNN



###### 基于深度学习



CNN，RNN，LSTM，GRU，Transformer，BERT



###### 文本分类的应用



情感分析，新闻分类，文本垃圾过滤，主题分类，实体识别，事件预测，分类搜索，法律文书分类，自动摘要和归类等



###### 文本分类的步骤



数据收集，数据预处理，特征工程，模型选择和训练，模型评估，模型优化和调参

![image-20260406221639094](D:\笔记\assets\image-20260406221639094.png)



#### 文本聚类



文本聚类是指将相似的文本分组或归类到同一个簇中



与文本分类不同，文本聚类不需要预先定义的类别，而是通过计算文本之间的相似度或距离来确定它们之间的相似关系



###### 常用的文本聚类算法



k-means算法，层次聚类算法，DBSCAN算法（具有噪声的基于密度的聚类方法），LDA模型



###### 文本聚类的应用



信息检索，文档组织与管理，主题发现与趋势分析，推荐系统，文本摘要，社交媒体分析，垃圾邮件过滤，生物信息学，客户反馈分析，教育资源分类等



###### 文本聚类的步骤



数据收集，数据预处理，特征提取，聚类算法选择，模型训练和聚类，结果评估和优化

![image-20260406222409548](D:\笔记\assets\image-20260406222409548.png)



#### 任务：垃圾短信分类



背景，某运营商已经积累大量的垃圾短信数据，共677291条数据，数据包括“短信ID”、“审核结果”和“短信文本内容”3列，“审核结果”列中0表示非垃圾短信，1表示垃圾短信



###### 数据读取



![image-20260406223102691](D:\笔记\assets\image-20260406223102691.png)



###### 数据预处理



![image-20260406223715542](D:\笔记\assets\image-20260406223715542.png)



###### 词频统计



![image-20260406224631061](D:\笔记\assets\image-20260406224631061.png)



###### 分类



![image-20260406225347114](D:\笔记\assets\image-20260406225347114.png)



###### 模型评价



![image-20260406225324560](D:\笔记\assets\image-20260406225324560.png)



#### 任务：新闻文本聚类



本次数据来自某新闻网站，该数据总共有15个类别标签，每个类别标签下分别有500条新闻数据，对新闻文本数据进行聚类



###### 数据读取



###### 文本预处理



###### 特征提取



###### 聚类



###### 模型评价



![image-20260406230108829](D:\笔记\assets\image-20260406230108829.png)



## 文本情感分析



#### 文本情感分析简介



文本情感分类是指对文本进行情感分类的任务，常见的情感类别包括积极、消极和中性三类



###### 文本情感分析的主要内容



文本情感分类，文本情感强度评估，文本情感目标识别，文本情感原因分析，情感时序分析

![image-20260407085610194](D:\笔记\assets\image-20260407085610194.png)



###### 文本情感分析的常见应用



舆情监测和管理，社交媒体分析，产品评论和用户体验分析，市场调研和竞争情报分析，在线客服和情感识别



#### 情感分析常用方法



###### 基于情感词典的分析方法



基于情感词典的分析方法是通过预先定义的情感词典来分析文本情感倾向的一种技术



情感词典是一个包含大量情感词及其情感倾向和可能的强度指标的集合，这种方法主要依赖于匹配文本中的词语与情感词典中的词语，从而判断整个文本的情感倾向

![image-20260407093434303](D:\笔记\assets\image-20260407093434303.png)



###### 机器学习方法



机器学习方法在情感分析中被广泛应用，常见的机器学习情感分析方法有朴素贝叶斯分类器、SVM、决策树和随机森林



这些机器学习方法都需要有标注的训练数据进行模型的训练和优化，在实际应用中，可以尝试多种方法的组合，甚至引入预训练的语言模型，以获得更好的情感分析效果

![image-20260407093735993](D:\笔记\assets\image-20260407093735993.png)



###### 深度学习方法



深度学习方法因其在特征学习和模型表达能力上的优势，被广泛应用于文本情感分析



常用的深度学习方法有RNN、CNN、LSTM、GRU和Transformer

![image-20260407094103289](D:\笔记\assets\image-20260407094103289.png)



#### 任务：基于情感词典的情感分析



基于情感词典的文本情感分析首先对文本分词，找出文本中的情感词、否定词和程度副词



然后判断每个情感词的前面是否存在否定词及程度副词，将情感词和它之前的否定词和程度副词划分为一个组，如果有否定词那么将情感词的情感权值乘-1，如果有程度副词那么乘程度副词的程度值，最后对所有组的得分求和，大于0的归于正面，小于0的归于负面



在实际应用中，基于情感词典的情感分析对情感词典的依赖较大，对于程度副词词典和否定词词典的依赖也较大，在实际应用时，可以寻找内容更为完善的词典进行文本情感分析，使得分析更加准确

![image-20260407103117406](D:\笔记\assets\image-20260407103117406.png)



#### 任务：基于机器学习的情感分析



###### 基于朴素贝叶斯分类的文本情感分析



随着互联网的普及，社交媒体和在线平台上的文本数据呈爆炸式增长，情感分析作为一种自然语言处理技术，旨在从这些文本中自动识别和提取情感倾向，朴素贝叶斯分类器因其简单高效、对数据量要求低等优点，成为情感分析中常用的算法之一

![image-20260407103550394](D:\笔记\assets\image-20260407103550394.png)



###### 基于snownlp的文本情感分析



snownlp库没有使用NLTK，所有的算法都是作者自己编写实现的，并且自带一些训练好的字典



snownlp库的主要功能有中文分词、词性标注、情感分析、文本分类、转换成拼音、繁体转简体、提取文本关键词、提取文本摘要、分割成句子等（https://pypi.org/project/snownlp/）

![image-20260407103936556](D:\笔记\assets\image-20260407103936556.png)



## NLP中的深度学习技术



#### RNN概述



###### RNN是什么



RNN（Recurrent Neural Network）是一种专门用于处理序列数据的神经网络架构，它通过在网络的隐藏层引入循环机制，使得网络能够保持对信息的记忆，从而在处理当前输入时考虑到历史信息



因为语言本质上是序列化的数据，因此RNN特别适合用于NLP领域



RNN的核心思想是使用循环结构来处理序列数据，其中每个序列元素都会按顺序进入网络，在处理每个元素时，网络不仅会考虑当前输入，还会综合之前的信息，这是通过将隐藏层的输出作为下一时间步的额外输入来实现的，从而形成了一种“记忆”机制，使网络能够捕捉到时间序列中的动态变化



###### RNN的发展



RNN的研究和应用时间较长，早期的RNN由于梯度消失和梯度爆炸问题，在长序列上的性能欠佳



LSTM（1997年）的提出有效解决了这一问题，通过引入门控机制（遗忘门、输入门和输出门）来控制信息的流动，极大地提高了模型对长期依赖的捕捉能力



此后，GRU作为一种简化的LSTM变体出现，解决了梯度问题的同时，也减少了模型的参数量



#### RNN结构



RNN的结构主要由输入层、隐藏层和输出层组成

![image-20260324175108870](D:\笔记\assets\image-20260324175108870.png)



输入层：接受序列数据的输入 $X$，可以是一个或多个特征向量



隐藏层：RNN的核心部分，负责处理序列数据并保持状态信息。隐藏状态在不同时间步之间共享参数（权重矩阵 $U、V、W$），使得网络可以处理不同长度的序列，并且能够推广到未知长度的序列



循环连接：隐藏层 $h$ 中的神经元通过循环连接将前一个时间步的隐藏状态传递给当前时间步



输出层：根据隐藏层的状态计算输出结果 $y$，输出层可以是一个或多个神经元，具体取决于任务的需求



RNN按照输入/输出的序列长度可以划分为多对一、等长的多对多、非等长三种结构



###### 多对一结构



RNN的多对一结构指输入序列经过RNN处理后，最终输出一个结果



这种结构非常适合于需要从变长输入序列中提取信息并产生单个输出的任务，比如输入一段评论，输出情感极性（正面/负面）



###### 等长的多对多结构



RNN的等长多对多结构适合于那些输入序列和输出序列长度相同的任务



基本结构包括三个主要部分：输入层、RNN层（可能包括多个RNN单元堆叠而成），以及输出层



输入层，负责将输入数据（通常是词或字符）转换为机器可处理的形式，如词嵌入向量



RNN层，这是模型的核心，由一系列的循环单元组成（可以是简单的RNN单元、LSTM或GRU单元）



输出层，在每个时间步，基于当前的隐藏状态，RNN会输出一个向量，这个向量随后被转换成最终的输出，如一个标签序列



双向RNN的改进之处在于不仅从前往后保留该词前面的词的信息，而且从后往前去保留该词后面的词的信息，然后基于这些信息进行预测该词，例如，如果预测一个语句中缺失的词语，那么需要根据上下文来进行预测



多层RNN具有更强大的表达与学习能力，但是复杂性也提高了，同时需要更多的训练数据



LSTM网络通过梯度剪裁技术克服梯度爆炸问题，当计算的梯度超过阈值c或者小于阈值-c的时候，便把此时的梯度设置成c或-c



###### 非等长结构



RNN的非等长结构适合处理那些要求输入数据和输出数据长度不相等的任务



最基础的Seq2Seq模型包含3个部分，即编码器（Encoder）、解码器（Decoder）和连接两者的语义向量C



注意力机制通过允许Decoder在生成每个目标词时“注意”Encoder的不同部分，从而可以根据需要动态地调整对输入序列的关注程度



由于Encoder-Decoder结构没有输入和输出等长的限制，因此Seq2Seq模型应用的范围非常广泛，比如机器翻译，文本摘要，阅读理解，语音识别等



#### 任务：基于LSTM的文本分类与情感分析



LSTM（Long Short-Term Memory）



###### 文本分类



文本分类模型采用两层LSTM模型，训练数据集来自THUCNews



THUCNews是一个由清华大学提供的中文新闻数据集，包含大量的中文新闻文本数据

![image-20260328215323897](D:\笔记\assets\image-20260328215323897.png)



###### 情感分析



数据来源为某电商平台的热水器评论数据，其中正面评论10679条，负面评论10428条



通过构建LSTM模型识别各条评论的情感倾向，即进行情感分析，正面评论标签为1，负面评论标签为0

![image-20260327221713333](D:\笔记\assets\image-20260327221713333.png)



#### 任务：基于Seq2Seq的机器翻译



![image-20260327143701380](D:\笔记\assets\image-20260327143701380.png)



## 智能问答系统



2014年，一个名为尤金·古斯特曼（Eugene Goostman）的聊天机器人成功地让人类相信它是一个13岁男孩



目前（2026年），智能问答系统是人工智能和NLP领域中一个备受关注并具有广阔发展前景的方向



#### 问答系统的主要组成



智能问答系统流程由问题理解、知识检索、答案生成3个部分组成



问题理解包括问题分类、关键词提取



知识检索包括结构化和非结构化信息检索



答案生成包括答案提取和答案验证



#### 任务：基于Seq2Seq的智能问答系统



###### 读取语料



![image-20260407163545882](D:\笔记\assets\image-20260407163545882.png)



###### 语料预处理



![image-20260407164740191](D:\笔记\assets\image-20260407164740191.png)



###### 模型构建



![image-20260407164816053](D:\笔记\assets\image-20260407164816053.png)



###### 模型训练



![image-20260407183917123](D:\笔记\assets\image-20260407183917123.png)



###### 模型评估



该环节在模型训练中就已经体现了（损失值）

![image-20260407170409372](D:\笔记\assets\image-20260407170409372.png)



为体现模型效果，套一层Flask框架，注意要有前端网页的文件

![image-20260407170821124](D:\笔记\assets\image-20260407170821124.png)



等待模型训练完成，运行flask框架，从浏览器去测试这个智障问答系统

![image-20260407175040753](D:\笔记\assets\image-20260407175040753.png)



## 大语言模型



#### 大语言模型简介



利用深度神经网络来对文本数据进行建模，并通过大规模预训练来获取通用的语言表示



#### 中文大语言模型



文心一言（百度），通义千问（阿里巴巴），混元（腾讯），盘古大模型（华为），讯飞星火认知大模型（科大讯飞），智谱清言（智谱华章），言犀（京东）



#### 讯飞星火认知大模型



注册后创建一个“自然语言处理”的应用，找到自己的API Key

![image-20260322230912362](D:\笔记\assets\image-20260322230912362.png)



简单使用一下API

![image-20260322232526438](D:\笔记\assets\image-20260322232526438.png)



#### 讯飞API程序开发应用



首先在config.json文件里配置一些参数



###### 情感分析



![image-20260323205550703](D:\笔记\assets\image-20260323205550703.png)



![image-20260323205746269](D:\笔记\assets\image-20260323205746269.png)



###### 文本分类



![image-20260323205911770](D:\笔记\assets\image-20260323205911770.png)



###### 机器翻译



![image-20260323210015329](D:\笔记\assets\image-20260323210015329.png)



###### 语义相似度计算



![image-20260323210320525](D:\笔记\assets\image-20260323210320525.png)



###### 关键词提取



![image-20260323210358882](D:\笔记\assets\image-20260323210358882.png)



###### 命名实体识别



![image-20260323210511983](D:\笔记\assets\image-20260323210511983.png)



###### 自动摘要



![image-20260323210659680](D:\笔记\assets\image-20260323210659680.png)



###### 文本纠错



![image-20260323210815767](D:\笔记\assets\image-20260323210815767.png)



###### 对话系统构



![image-20260323211443469](D:\笔记\assets\image-20260323211443469.png)



## ~~基于TipDM大数据挖掘建模平台实现垃圾短信分类~~



网址：https://python.tipdm.org/auth/register.jspx

![image-20260323214053785](D:\笔记\assets\image-20260323214053785.png)



由于资金等原因，该章节不进行复现

![image-20260407163000292](D:\笔记\assets\image-20260407163000292.png)



## 尾部标记
