import os
import jieba
# from tkinter import _flatten
# import json


# 代码11-1 读取语料库文件
def read_corpus(corpus_path):
    # 列出文件路径下所有文件
    corpus_files = os.listdir(corpus_path)
    corpus = []
    # 循环读取各个文件内容
    for corpus_file in corpus_files:
        with open(os.path.join(corpus_path, corpus_file), 'r', encoding='utf-8') as f:
            corpus.extend(f.readlines())
    corpus = [i.replace('\n', '') for i in corpus]
    # 返回语料库的列表数据
    return corpus


corpus = read_corpus(corpus_path='data/dialog/')
print('语料库读取完成！'.center(30, '='))
print('语料库展示: \n', corpus[:6])


# 代码11-2 分词并构建词典
def word_cut(corpus, user_dict):
    '''
    corpus：语料
    user_dict：自定义词典
    '''
    # 加载自定义词典
    jieba.load_userdict(user_dict)
    corpus_cut = [jieba.lcut(i) for i in corpus]

    print('分词完成'.center(30, '='))
    return corpus_cut


def _flatten(seq):
    """自己写一个拉平函数替代 tkinter"""
    result = []
    for item in seq:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


# 构建词典
def get_dict(corpus_cut):
    '''
    corpus_cut：分词后的语料文件
    '''
    # 将分词结果列表拉直
    tmp = _flatten(corpus_cut)
    # 去除重复词，保留所有出现的唯一的词
    all_dict = list(set(tmp))
    id2words = {i: j for i, j in enumerate(all_dict)}
    # 构建词典
    words2id = dict(zip(id2words.values(), id2words.keys()))
    print('词典构建完成'.center(30, '='))
    return all_dict, id2words, words2id


# 执行分词
corpus_cut = word_cut(corpus, user_dict='data/ids/mydict.txt')
print('分词结果展示: \n', corpus_cut[:2])
# 获取字典
all_dict, id2words, words2id = get_dict(corpus_cut)
print('词典展示: \n', all_dict[:6])


# 代码11-3 拆分问、答和保存文件
def save(all_dict, corpus_cut, file_path):
    '''
    all_dict: 获取的词典
    file_path: 文件保存路径
    corpus_cut: 分词后的语料文件
    '''
    if not os.path.exists(file_path):
        os.makedirs(file_path)  # 如果文件夹不存在则新建
    source = corpus_cut[::2]  # 问
    target = corpus_cut[1::2]  # 答
    # 构建文件的对应字典
    file = {'all_dict.txt': all_dict, 'source.txt': source, 'target.txt': target}
    # 分别进行文件处理并保存
    for i in file.keys():
        if i in ['all_dict.txt']:
            with open(os.path.join(file_path, i), 'w', encoding='utf-8') as f:
                f.writelines(['\n'.join(file[i])])
        else:
            with open(os.path.join(file_path, i), 'w', encoding='utf-8') as f:
                f.writelines([' '.join(i) + '\n' for i in file[i]])
print('文件已保存'.center(30, '='))

# 执行保存
save(all_dict, corpus_cut, file_path='output')

