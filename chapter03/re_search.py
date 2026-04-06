# 代码3-3 通过search函数进行匹配
import re

text1 = ('自然语言处理是研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。'
         '自然语言处理是一门融语言学、计算机科学、数学于一体的科学。')
print(re.search('通信', text1))

# 代码3-4 通过search函数获取子串匹配
# 在字符串中搜索匹配的子串
result = re.search(r'(\d+)-(\d+)-(\d+)', '2023-10-30')
if result:
    # 获取整个匹配到的子串
    print(result.group())  # 输出：2023-10-30
    # 获取第一个括号内的子串
    print(result.group(1))  # 输出：2023
    # 获取第二个括号内的子串
    print(result.group(2))  # 输出：10
    # 获取第三个括号内的子串
    print(result.group(3))  # 输出：30
