import requests
from bs4 import BeautifulSoup
import jieba
from opencc import OpenCC
from collections import Counter
import re

# 从网页获取中文文本数据  # 红楼梦
url = "https://www.gutenberg.org/cache/epub/24264/pg24264-images.html"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    response = requests.get(url, headers=headers)
    # 检查状态码
    if response.status_code == 200:
        # 打印前100个字符测试
        print(response.text[:100])
        html = response.text
    else:
        print(f"请求失败，状态码: {response.status_code}")
except Exception as e:
    print(f"发生错误: {e}")

# 使用BeautifulSoup提取文本内容
soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text()

# 清洗文本，移除非中文字符
text = re.sub(r'[^\u4e00-\u9fa5。,，！!?？]', '', text)

# 繁体转简体
cc = OpenCC('t2s')
text_simplified = cc.convert(text)

# 使用jieba进行分词
words = jieba.cut(text_simplified)
# 加载停用词
# 可以添加自己的停用词列表
stop_words = set()
with open('data/stopwords.txt', 'r', encoding='gbk') as f:
    for word in f:
        stop_words.add(word.strip())
filtered_words = [word for word in words if word.strip() and word.strip() not in stop_words]

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


# 查询词频在指定区间内的词数量
def count_words_in_range(word_counts, min_freq, max_freq):
    return sum(min_freq <= count <= max_freq for count in word_counts.values())

min_freq = 5
max_freq = 10
range_count = count_words_in_range(word_counts, min_freq, max_freq)
print(f"\n词频在 {min_freq} 到 {max_freq} 之间的词的数量：{range_count}")


# 预览部分文本
def preview_text(text, start, end):
    print("\n预览部分文本:")
    print(text[start:end])

preview_text(text_simplified, 100, 200)
