import requests
from bs4 import BeautifulSoup
from opencc import OpenCC

# 定义繁体转简体的函数
def traditional_to_simplified(text):
    # 繁体转简体
    cc = OpenCC('t2s')
    return cc.convert(text)

# 获取网页内容  # 《西游记》网址
url = "https://www.gutenberg.org/cache/epub/23962/pg23962-images.html"
response = requests.get(url)
# 确保正确的编码
response.encoding = 'utf-8'
# 解析网页内容
soup = BeautifulSoup(response.text, 'html.parser')
# 假设我们想获取<body>标签内的所有文本
body_text = soup.body.get_text()
# 将繁体转换为简体
simplified_text = traditional_to_simplified(body_text)

# 将文本分割为行并取100-105行查看
lines = simplified_text.splitlines()
selected_lines = lines[100:105]
for line in selected_lines:
    print(line)

