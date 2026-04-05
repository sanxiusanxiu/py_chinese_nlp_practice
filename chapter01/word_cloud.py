import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# 读取文本文件
with open("input/二十大报告.txt", "r", encoding="utf-8") as f:
    text = f.read()
# 读取中文停用词表
with open("input/stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = f.read()

# 分词
seg_list = jieba.cut(text)
words = " ".join(seg_list)

# 设置停用词，用于过滤一些常见无意义的词语
def remove_stopwords(text, stopwords):
    text_without_stopwords = [word for word in re.split(r'\W+', text) if word not in stopwords]
    return ' '.join(text_without_stopwords)

text_without_stopwords = remove_stopwords(words, stopwords)

# 统计词频：使用Python中的`collections`库统计每个词语出现的频率
word_count = Counter(text_without_stopwords.split())
# 统计出现频率最高的前10个词汇
n = 10
top_words = word_count.most_common(n)

# 打印频率最高的前10个词汇
for word, count in top_words:
    print(f"词语: {word}  频次: {count}")

# 生成词云图：使用`Wordcloud`库根据词频数据生成词云图。可以设置不同的参数来调整词云图的外观，如背景颜色、字体样式、词语显示大小等。
wordcloud = WordCloud(background_color='white',
                      font_path='QingNiaoHuaGuangJianMeiHei-2.ttf',
                      scale=2,
                      stopwords=stopwords).generate_from_frequencies(word_count)
plt.figure(dpi=600)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
