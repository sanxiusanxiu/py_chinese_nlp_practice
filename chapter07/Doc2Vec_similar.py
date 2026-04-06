# Doc2Vec计算文本相似度
import sys
import re
import jieba
import codecs
import gensim
import numpy as np
import pandas as pd


def segment(doc: str):
    stop_words = pd.read_csv('data/stopwords.txt', index_col=False, quoting=3,
                             names=['stopword'], sep='\t', encoding='utf-8-sig')
    stop_words = list(stop_words.stopword)
    reg_html = re.compile(r'<[^>]+>', re.S)  # 去掉html标签数字等
    doc = reg_html.sub('', doc)
    doc = re.sub('[０-９]', '', doc)
    doc = re.sub('\s', '', doc)
    word_list = list(jieba.cut(doc))
    out_str = ''
    for word in word_list:
        if word not in stop_words:
            out_str += word
            out_str += ' '
    segments = out_str.split(sep=' ')
    return segments


def doc2vec(file_name, model):
    start_alpha = 0.01
    infer_epoch = 1000
    doc = segment(codecs.open(file_name, 'r', 'utf-8').read())
    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, epochs=infer_epoch)
    return doc_vec_all


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


def test_model(file1, file2):
    print('导入模型')
    model_path = 'output/zhwk_news.doc2vec'
    model = gensim.models.Doc2Vec.load(model_path)
    vect1 = doc2vec(file1, model)  # 转成句子向量
    vect2 = doc2vec(file2, model)
    print(sys.getsizeof(vect1))  # 查看变量占用空间大小
    print(sys.getsizeof(vect2))
    cos = similarity(vect1, vect2)
    print('相似度：%0.2f%%' % (cos * 100))


if __name__ == '__main__':
    file1 = 'data/corpus_test/t1.txt'
    file2 = 'data/corpus_test/t2.txt'
    test_model(file1, file2)
