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
    path = 'data/trainCorpus.txt'
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
