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
