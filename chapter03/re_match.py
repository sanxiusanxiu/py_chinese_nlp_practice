# 代码3-1 使用match函数匹配文本
import re

text1 = ('自然语言处理是研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。'
         '自然语言处理是一门融语言学、计算机科学、数学于一体的科学。')
print('匹配的结果是：', re.match('自然语言处理', text1))
print('匹配的结果是：', re.match('语言处理', text1))

# 代码3-2 对文本text1进行切分后再进行匹配
# 以句号为分隔符通过split切分text1
p_string = text1.split('。')
for line in p_string:
    # 查找当前行是否匹配“自然语言处理”，如果匹配到，那么打印这行信息
    if re.match('自然语言处理', line) is not None:
        print(line)