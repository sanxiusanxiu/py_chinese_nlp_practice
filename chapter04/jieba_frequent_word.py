import requests
from bs4 import BeautifulSoup
import jieba
from opencc import OpenCC
from collections import Counter
import re

# 读取新闻文本
with open("data/flightnews.txt", "r", encoding="utf-8") as f:
    text = f.read()
print("文本长度:", len(text))

# 加载停用词
stop_words = set()
with open('data/stopword.txt', 'r', encoding='utf-8') as f:
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
