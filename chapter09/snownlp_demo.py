# 代码9-11 Snownlp中调用函数进行情感分析
from snownlp import SnowNLP

# 创建snownlp对象，设置要测试的语句
s1 = SnowNLP('这东西真的挺不错的')
s2 = SnowNLP('垃圾东西')
print('调用sentiments方法获取s1的积极情感概率为:', s1.sentiments)
print('调用sentiments方法获取s2的积极情感概率为:', s2.sentiments)
