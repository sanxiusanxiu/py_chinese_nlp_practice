# 代码9-6 词典情感分析
import jieba

# 情感词典（正面情感词和负面情感词）
positive_words = ['喜欢', '愉快', '赞美', '好']
negative_words = ['讨厌', '悲伤', '垃圾', '坏']


def sentiment_analysis(text):
    # 分词
    words = jieba.lcut(text)
    # 初始化情感得分
    sentiment_score = 0
    # 遍历每个词
    for word in words:
        if word in positive_words:
            sentiment_score += 1  # 正面情感得分加1
        elif word in negative_words:
            sentiment_score -= 1  # 负面情感得分减1
    # 根据得分判断情感极性
    if sentiment_score > 0:
        sentiment = '正面'
    elif sentiment_score < 0:
        sentiment = '负面'
    else:
        sentiment = '中性'
    return sentiment


# 测试例子
text = '这部电影太好看了，赞美一下！'
result = sentiment_analysis(text)
print('情感极性：', result)
text = '这家餐厅的服务太差了，讨厌！'
result = sentiment_analysis(text)
print('情感极性：', result)
