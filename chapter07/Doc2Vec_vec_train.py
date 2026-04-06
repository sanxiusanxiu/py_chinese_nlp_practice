import os
import time

import jieba
import gensim
from opencc import OpenCC
from gensim.corpora import WikiCorpus


# 定义LabeledLineSentence类，进行数据预处理
class LabeledLineSentence(object):
    def __init__(self, wkc):
        self.wkc = wkc
        # 获取文章标题作为Tag
        self.wkc.metadata = True

    def __iter__(self):
        cc = OpenCC('t2s')
        for content, (page_id, title) in self.wkc.get_texts():
            words = [w for c in content for w in jieba.cut(cc.convert(c))]
            yield gensim.models.doc2vec.TaggedDocument(words=words, tags=[title])


# 使用Doc2Vec训练段落向量
def train():
    zh_name = 'data/zh-latest-pages-articles.xml.bz2'
    print('正在加载维基百科语料 (仅加载结构，不加载全量文本)...')
    wkc = WikiCorpus(zh_name, dictionary={})
    documents = LabeledLineSentence(wkc)
    print('开始训练 Doc2Vec 模型...')
    start_time = time.time()
    # dbow_words设为0，表示不同时训练词向量，防止本地内存爆炸
    model = gensim.models.Doc2Vec(documents, dm=0, dbow_words=1, vector_size=192,
                                  window=8, min_count=10, epochs=5, workers=4)
    model.save('output/zhwk_news.doc2vec')


def test_model():
    try:
        print('正在加载模型...')
        model = gensim.models.Doc2Vec.load('output/zhwk_news.doc2vec')
        print('\n--- 词向量相似度测试 (由Doc2Vec顺带学习到的) ---')
        try:
            print(f'"番茄" 与 "西红柿" 的相似度: {model.wv.similarity("番茄", "西红柿"):.4f}')
        except KeyError:
            print('词库中缺少 "番茄" 或 "西红柿" (因为 dbow_words=0，词向量精度可能较低)')
        try:
            print(f'"货车" 与 "卡车" 的相似度: {model.wv.similarity("货车", "卡车"):.4f}')
        except KeyError:
            print('词库中缺少 "货车" 或 "卡车"')
        print('\n--- 段落向量相似度测试 (Doc2Vec的核心功能) ---')
        # 提示：因为你之前用文章标题做了tags，所以你可以直接用标题找相似文章
        try:
            similar_docs = model.dv.most_similar('人工智能', topn=3)
            print(f'与 "人工智能" 最相似的维基百科条目：')
            for title, score in similar_docs:
                print(f'  - {title} (相似度: {score:.4f})')
        except KeyError:
            print('词库中没有找到 "人工智能" 这个标题的文档')
    except FileNotFoundError:
        print('错误：找不到模型文件，请先训练模型！')


if __name__ == '__main__':
    if os.path.exists('output/zhwk_news.doc2vec') == False:
        print('开始训练模型')
        train()
        print('模型训练完毕')

    test_model()
