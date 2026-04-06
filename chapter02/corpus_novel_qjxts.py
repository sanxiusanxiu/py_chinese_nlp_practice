import jieba
from collections import Counter
import re
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取作品集文本
with open("data/七剑下天山.TXT", "r", encoding="gbk") as f:
    text = f.read()

# 统计文本的长度
print("文本长度:", len(text))

# 使用字符串切片方法查看部分文本
print("\n预览部分文本:")
print(text[100:300])

# 清洗文本，移除非中文字符
text_cleaned = re.sub(r'[^\u4e00-\u9fa5。,，！!?？]', '', text)

# 使用jieba进行分词
words = jieba.cut(text_cleaned)

# 加载停用词
stop_words = set()
try:
    with open('data/stopwords.txt', 'r', encoding='gbk') as f:
        for word in f:
            stop_words.add(word.strip())
except FileNotFoundError:
    print("停用词文件未找到，将不使用停用词过滤")

# 过滤停用词
filtered_words = [word for word in words if word.strip() and word.strip() not in stop_words]

# 统计文本中的词语频数分布
word_counts = Counter(filtered_words)

# 统计词频
print("\n文本总词数：", len(filtered_words))
print("不重复词汇总数：", len(word_counts))

# 输出高频词统计
top_n = 15
print(f"\n出现频率最高的{top_n}个词：")
for word, count in word_counts.most_common(top_n):
    print(word, ":", count)

# 查看指定词的上下文
def find_context(text, word, window=50):
    contexts = []
    pattern = re.compile(f'.{{0,{window}}}{re.escape(word)}.{{0,{window}}}', re.DOTALL)
    matches = pattern.findall(text)
    for match in matches[:5]:  # 只返回前5个匹配
        contexts.append(match.strip())
    return contexts

# 示例：查找"天山"的上下文
keyword = "天山"
print(f"\n\"{keyword}\"的上下文：")
contexts = find_context(text, keyword)
for i, context in enumerate(contexts):
    print(f"{i+1}. {context}")

# 搜索相似词语
def find_similar_words(word_counts, prefix, top_n=10):
    similar_words = [(word, count) for word, count in word_counts.items() if word.startswith(prefix)]
    similar_words.sort(key=lambda x: x[1], reverse=True)
    return similar_words[:top_n]

# 示例：搜索以"剑"开头的词语
prefix = "剑"
print(f"\n以\"{prefix}\"开头的高频词语：")
similar_words = find_similar_words(word_counts, prefix)
for word, count in similar_words:
    print(word, ":", count)

# 绘制离散图
plt.figure(figsize=(12, 6))
top_words = [word for word, _ in word_counts.most_common(20)]
top_counts = [count for _, count in word_counts.most_common(20)]

plt.bar(range(len(top_words)), top_counts)
plt.xticks(range(len(top_words)), top_words, rotation=45, ha='right')
plt.xlabel('词语')
plt.ylabel('出现次数')
plt.title('《七剑下天山》高频词统计')
plt.show()
plt.tight_layout()
# plt.savefig('output/qjxts_word_frequency.png')
# print("\n高频词统计图表已保存到 output/qjxts_word_frequency.png")

# 保存语料库
with open('output/qjxts_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(text_cleaned)
print("\n《七剑下天山》语料库已保存到 output/qjxts_corpus.txt")
