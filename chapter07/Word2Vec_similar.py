# Word2Vec计算文本相似度
import sys
import codecs  # 加载codecs库用于编码转换
import gensim
import numpy as np
import pandas as pd
from jieba import analyse


# 使用TF-IDF算法，提取句子的关键词
def keyword_extract(data):
    tfidf = analyse.extract_tags
    keywords = tfidf(data)
    return keywords


# 将文档中的每句话进行关键词提取并保存
def segment(file, keyfile):
    with open(file, 'r', encoding='utf-8') as f, open(
            keyfile, 'w', encoding='utf-8') as k:
        for doc in f:
            keywords = keyword_extract(doc[:len(doc) - 1])
            for word in keywords:
                k.write(word + ' ')
            k.write('\n')


# 获取字符串中某字符的位置以及出现的总次数
def get_char_pos(string, char):
    chPos = []
    try:
        chPos = list(((pos) for pos, val in enumerate(string) if (val == char)))
    except:
        pass
    return chPos


# 获取关键词的词向量
def word2vec(file_name, model):
    wordvec_size = 192  # 词向量的维度
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        word_vec_all = np.zeros(wordvec_size)  # 生成包含192个元素的零矩阵
        for data in f:
            space_pos = get_char_pos(data, ' ')
            first_word = data[0: space_pos[0]]
            if model.wv.__contains__(first_word):
                word_vec_all = word_vec_all + model.wv[first_word]
            for i in range(len(space_pos) - 1):
                word = data[space_pos[i]: space_pos[i + 1]]
                if model.wv.__contains__(word):  # 判断模型是否包含该词语
                    word_vec_all = word_vec_all + model.mv[word]
        return word_vec_all


# 计算两个向量余弦值
def similarity(a_vect, b_vect):
    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = None
    for a, b in zip(a_vect, b_vect):
        dot_val += a * b
        a_norm += a ** 2
        b_norm += b ** 2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm * b_norm) ** 0.5)
    return cos


def test_model(keyfile1, keyfile2):
    print('导入模型')
    model_path = 'output/zhwk_news.word2vec'
    model = gensim.models.Word2Vec.load(model_path)  # 加载模型
    vect1 = word2vec(keyfile1, model)
    vect2 = word2vec(keyfile2, model)
    print(sys.getsizeof(vect1))  # 查看变量占用空间大小
    print(sys.getsizeof(vect2))
    cos = similarity(vect1, vect2)
    print('相似度：%0.2f%%' % (cos * 100))


# 计算文本的相似度
if __name__ == '__main__':
    file1 = 'data/corpus_test/t1.txt'
    file2 = 'data/corpus_test/t2.txt'
    keyfile1 = 'data/corpus_test/t1_key.txt'
    keyfile2 = 'data/corpus_test/t2_key.txt'
    segment(file1, keyfile1)
    segment(file2, keyfile2)
    test_model(keyfile1, keyfile2)
    file1 = 'data/corpus_test/t1_key.txt'
