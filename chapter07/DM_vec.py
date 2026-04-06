# 代码7-6 DM模型实现文本向量化
import jieba
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 准备数据：定义待处理的句子列表
sentences = [
    '中文NLP',
    'DM模型用于文本表示和语义建模',
    '文本分类任务需要将句子转换为向量表示'
]
# 分词：使用jieba进行中文分词
tokenized_sentences = [jieba.lcut(sentence) for sentence in sentences]
# 构建标记文档：为每个分词后的句子分配一个唯一的标签
tagged_documents = [
    TaggedDocument(words=words, tags=[str(i)])
    for i, words in enumerate(tokenized_sentences)
]
# 定义Doc2Vec模型（DM）并训练
# vector_size: 向量维度
# window: 上下文窗口大小
# min_count: 忽略出现次数低于min_count的词
# dm: 训练算法选择(1=DM, 0=DBoW)
model = Doc2Vec(vector_size=100, window=5, min_count=1, dm=1)

# 建立模型词汇表
model.build_vocab(tagged_documents)
# 训练模型
model.train(tagged_documents, total_examples=model.corpus_count, epochs=10)
# 获取句子向量表示：将分词后的句子转换为向量
sentence_vectors = [model.infer_vector(words) for words in tokenized_sentences]
# 应用模型：将新的句子转换为向量表示
new_sentence = '新的句子'
new_tokenized_sentence = jieba.lcut(new_sentence)
new_sentence_vector = model.infer_vector(new_tokenized_sentence)
# 输出结果
for i, sentence_vector in enumerate(sentence_vectors):
    print(f"句子{i + 1}的向量表示: {sentence_vector}")
print(f"新句子的向量表示: {new_sentence_vector}")
