# 代码3-21 查看西游记的部分文本内容
import re
from urllib.request import urlopen
from opencc import OpenCC
url1 = 'https://www.gutenberg.org/cache/epub/23962/pg23962.html'
html1 = urlopen(url1).read()
text4 = html1.decode('utf-8')
text_temp = text4[7406:7699]  # 查看部分内容
# 创建一个简体中文转换器
converter = OpenCC('t2s')
# 转换为简体中文
simplified_text = converter.convert(text_temp)
print(simplified_text)
print('--------------------')
# 过滤掉所有英文字符、数字及英文特殊符号
print(re.sub('[\[\]\s+\.\!\/_,$%^*(+\"\'?:&@#;<>=-]+|[a-zA-Z]+|[0-9]+', '', simplified_text))
