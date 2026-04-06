import pandas as pd
import re


def preprocess_text(text):
    # 清洗文本：去除标点符号、数字和多余空格
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_corpus(input_file, output_file):
    # 从CSV文件中读取数据
    df = pd.read_csv(input_file)
    # 提取评论文本
    comments = df['comment'].dropna().tolist() if 'comment' in df.columns else []
    # 预处理评论文本
    preprocessed_comments = [preprocess_text(comment) for comment in comments]
    # 将评论文本写入到输出文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        for comment in preprocessed_comments:
            # 确保评论文本非空
            if comment:
                f.write(comment + '\n')

input_file = 'data/ratings.csv'
output_file = 'output/movie_comments.txt'
build_corpus(input_file, output_file)
print("电影评论语料库已建立")

# 随机选择一个评论进行显示
import random

def get_random_comment(corpus_file):
    with open(corpus_file, 'r', encoding='utf-8') as f:
        comments = f.readlines()
        random_comment = random.choice(comments)
        return random_comment.strip()


corpus_file = 'output/movie_comments.txt'
random_comment = get_random_comment(corpus_file)
print("随机选择的评论：", random_comment)
