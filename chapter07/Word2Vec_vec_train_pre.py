from gensim.corpora import WikiCorpus
import jieba
from opencc import OpenCC

# 未测试
def reduce_zh():
    zh_name = 'data/zh-latest-pages-articles.xml.bz2'
    output_path = 'output/reduce_zh.txt'
    # 在循环外部只初始化一次 OpenCC 引擎
    cc = OpenCC('t2s')
    i = 0
    #
    with open(output_path, 'w', encoding='utf-8') as f:
        # 处理从xml文件中读出的训练语料
        wkc = WikiCorpus(zh_name, dictionary={})
        print("开始处理维基百科语料...")
        for text in wkc.get_texts():
            for temp_sentence in text:
                # 跳过空行
                if not temp_sentence.strip():
                    continue
                # 繁体转简体
                simp_sentence = cc.convert(temp_sentence)
                # 利用jieba库对语料库中的句子进行分词
                seg_list = list(jieba.cut(temp_sentence))
                # 将这一句话的词用空格连接，并写入文件（一句话就是一行）
                f.write(' '.join(seg_list) + '\n')
            i = i + 1
            if i % 200 == 0:
                print(f'已处理并保存 {i} 篇文章')
    print(f"语料预处理完成，保存在: {output_path}")

# 语料预处理
if __name__ == '__main__':
    reduce_zh()
