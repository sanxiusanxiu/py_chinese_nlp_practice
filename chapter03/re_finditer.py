# 代码3-7 使用finditer迭代搜索
import re

text1 = ('自然语言处理是研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。'
         '自然语言处理是一门融语言学、计算机科学、数学于一体的科学。')
# 定义要匹配的正则表达式模式
pattern = r"自然语言处理"
# 使用re.finditer函数进行迭代搜索
matches = re.finditer(pattern, text1)
# 遍历匹配结果并打印每个匹配的起始位置和结束位置
for match in matches:
    print("匹配文本：", match.group())
    print("匹配起始位置：", match.start())
    print("匹配结束位置：", match.end())
    print("--------------------")