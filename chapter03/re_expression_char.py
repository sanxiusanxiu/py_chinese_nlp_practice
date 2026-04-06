# 代码3-9 量词元字符常见用法
import re

text1 = ('自然语言处理是研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。'
         '自然语言处理是一门融语言学、计算机科学、数学于一体的科学。')
# 唐初著名诗人刘希夷的诗《代悲白头翁 / 白头吟》中截取和拼接的两句
text2 = '今年花落颜色改，明年花开复谁在？年年岁岁花相似，岁岁年年人不同。'
re.findall('年?', text2)  # “年”最多重复1次
print(re.findall('年*', text2))  # “年”可以重复0或多次
re.findall('年+', text2)  # “年”可以重复1次或多次
re.findall('年{1}', text2)  # “年”正好被重复1次
re.findall('年{2}', text2)  # “年”正好被重复2次
re.findall('年{0,1}', text2)  # “年”至少重复0次，至多重复1次
re.findall('年.+', text2)  # 以年开始,后面可以跟任意多个字符
re.findall('年+.', text2)  # “年”可以重复1次或多次，后面跟任意字符
re.findall('年.?', text2)  # “年”后面至多可以跟1个任意字符
re.findall('年.*', text2)  # “年”后面可以跟任意多个字符
re.findall('年.+?', text2)  # “年”后面可以跟一个任意字符，并且这两个字符最多重复1次
re.findall('年.*?', text2)  # “年”后面允许不带其他字符的内容。
re.findall('年?花', text2)  # “花”前面的“年”最多重复1次
re.findall('年*花', text2)  # “花”前面的“年”可以重复0或多次
re.findall('年+花', text2)  # “花”前面的“年”可以重复1次或多次
re.findall('年{1}花', text2)  # “花”前面的“年”重复1次
re.findall('年{2}花', text2)  # “花”前面的“年”重复2次
re.findall('年{0,1}花', text2)  # “花”前面的“年”至少重复0次，至多重复1次
re.findall('年.+花', text2)  # “年”开头“花”结尾且中间的任意字符可以任意多个
re.findall('年.?花', text2)  # “年”开头“花”结尾且中间的任意字符至多一个
re.findall('年.*花', text2)  # “年”开头“花”结尾且中间的任意字符可以任意多个
re.findall('年.+?花', text2)  # “年”开头“花”结尾且中间至少带有一个字符的内容
re.findall('年.*?花', text2)  # “年”开头“花”结尾中间允许不带其他字符的内容

# 代码3-10 使用中括号“[ ]”进行匹配
print(re.findall('[科数]学', text1))  # 匹配[]内的任意一个字符

# 代码3-11 匹配所有以“自”开头的字符串
p_string = text1.split('。')
for line in p_string:
    if len(re.findall('^自', line)):
        print(line)

# 代码3-12 匹配所有以“学”为结尾的字符串
p_string = text1.split('、')
for line in p_string:
    if len(re.findall('学$', line)):
        print(line)

# 代码3-13 转义字符“\”的具体用法
text3 = 'Hello，everyone，我是/ 陈_X/ 我 的_/  、邮箱，地址是。 wxid_6cp@16.co'
re.sub('\\d', '数字', text3)  # 将text3中的阿拉伯数字替换为“数字”
re.sub(r'\d', '数字', text3)  # 将数字替换为“数字”
re.sub('[0-9]', '数字', text3)  # 将数字替换为“数字”
re.sub(r'\s', '', text3)  # 删除空白
re.sub(r'\w', '', text3)  # 删除字和数字
re.findall('\\b[a-zA-Z]+', text3)  # 查找带有多个英文字母的字符
re.findall('\\b[a-zA-Z]+\\b', text3)  # 查找只带有字母的单词

# 代码3-14 使用英文句号“.”进行匹配
print(re.findall('自.语言处理', text1))  # 匹配任意字符

# 代码3-15 使用管道符进行匹配
print(re.findall('方法|计算机', text1))

# 代码3-16 输出含有“方法”或“计算机”的句子
p_string = text1.split('。')
for line in p_string:
    if len(re.findall('方法|计算机', line)):
        print(line)
