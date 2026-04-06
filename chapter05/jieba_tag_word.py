# 代码5-3 基于jieba的词性标注
import jieba.posseg as pseg

text = "我爱北京天安门"
words = pseg.cut(text)
for word, flag in words:
    print(word, flag)
