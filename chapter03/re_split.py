# 代码3-8 使用split分割字符串
import re

text1 = ('自然语言处理是研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。'
         '自然语言处理是一门融语言学、计算机科学、数学于一体的科学。')
# 以空格、中文逗号和中文句号为分隔符
pattern = "[  ，。]"
result = re.split(pattern, text1)
print(result)
