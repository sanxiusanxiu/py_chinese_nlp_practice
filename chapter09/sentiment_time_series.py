# 代码9-5 中文情感时序分析
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import pandas as pd

# 中文文本数据
texts = [
    "今天心情很好，工作顺利",
    "周一了，不想上班啊",
    "开心的事情真是太少了",
    "听说明天要下雨，有点烦"
]
# 初始化情感分析结果列表
sentiments_sn = []

# 使用SnowNLP进行情感分析
for text in texts:
    sn = SnowNLP(text)
    # 使用SnowNLP计算情感得分，范围从0（负面）到1（正面）
    sentiments_sn.append(sn.sentiments)

# 构建DataFrame
df = pd.DataFrame({
    'Text': texts,
    'Sentiment_SN': sentiments_sn  # 存储使用SnowNLP分析的情感分析结果
})

# 输出情感分析结果
print(df)

# 绘制情感时序分析图
plt.figure(figsize=(10, 6), dpi=600)
plt.rcParams['font.family'] = ['SimHei']
plt.plot(df['Sentiment_SN'], marker='x', label='SnowNLP')
plt.xticks(ticks=range(len(df['Text'])), labels=df['Text'], rotation=45)
plt.xlabel('文本', fontsize=14)
plt.ylabel('情感得分', fontsize=14)
plt.title('中文情感时序分析', fontsize=16)
plt.legend()
plt.show()
