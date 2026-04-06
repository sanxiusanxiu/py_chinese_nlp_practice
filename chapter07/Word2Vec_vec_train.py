import os
import time
import gensim  # 加载gensim库自动提取文档语义主题
from gensim.models import Word2Vec  # 加载Word2Vec模块训练词向量
# 加载LineSentence库可以将类别加入到向量的训练中
from gensim.models.word2vec import LineSentence


def train():
    print('开始训练模型...')
    start_time = time.time()
    with open('output/reduce_zh.txt', 'r', encoding='utf-8') as wk_news:
        # 使用Word2Vec训练词向量
        model = Word2Vec(LineSentence(wk_news), sg=0, vector_size=192, window=5, min_count=5, workers=4)
    model.save('output/zhwk_news.word2vec')

    end_time = time.time()
    print(f'模型训练完毕，耗时: {end_time - start_time:.2f} 秒')


def test_model():
    try:
        print('正在加载模型...')
        model = gensim.models.Word2Vec.load('output/zhwk_news.word2vec')

        print('\n--- 相似度测试 ---')
        # 使用 try-except 防止因为词不在词典里而直接崩溃
        try:
            print(f'"番茄" 与 "西红柿" 的相似度: {model.wv.similarity("番茄", "西红柿"):.4f}')
        except KeyError:
            print('词库中缺少 "番茄" 或 "西红柿"')

        try:
            print(f'"货车" 与 "卡车" 的相似度: {model.wv.similarity("货车", "卡车"):.4f}')
        except KeyError:
            print('词库中缺少 "货车" 或 "卡车"')

    except FileNotFoundError:
        print('错误：找不到模型文件，请先训练模型！')


if __name__ == '__main__':
    # 确保输出目录存在（防止保存时报错）
    os.makedirs('output', exist_ok=True)
    # 如果没有模型就进行训练
    if not os.path.exists('output/zhwk_news.word2vec'):
        train()

    # 无论模型是刚训练完的，还是之前就有的，都执行测试
    test_model()
