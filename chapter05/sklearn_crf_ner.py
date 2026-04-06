# 5.5.2	命名实体识别流程
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


# 定义一个类来处理语料库
class CorpusProcess(object):
    def __init__(self):
        # 初始化
        self.train_corpus_path = "data/food.txt"
        self._maps = {u'n': u'NOUN', u'ns': u'LOC'}

    # 读取语料
    def read_corpus_from_file(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        return lines

    # 由词性提取标签
    def pos_to_tag(self, p):
        t = self._maps.get(p, None)  # 根据词性映射表提取对应的标签
        return t if t else u'O'

    # 标签使用BIO模式
    def tag_perform(self, tag, index):
        if index == 0 and tag != u'O':  # 如果是第一个词，且不是O标签，则使用B-标签
            return u'B_{}'.format(tag)
        elif tag != u'O':  # 如果是其他词，且不是O标签，则使用I-标签
            return u'I_{}'.format(tag)
        else:
            return tag

    # 初始化
    def initialize(self):
        lines = self.read_corpus_from_file(self.train_corpus_path)
        words_list = [line.strip().split(' ') for line in lines if line.strip()]
        del lines
        self.init_sequence(words_list)

    # 初始化序列
    def init_sequence(self, words_list):
        # 初始化字序列、词性序列、标记序列
        words_seq = [[word.split(u'/')[0] for word in words] for words in words_list]
        pos_seq = [[word.split(u'/')[1] for word in words] for words in words_list]
        tag_seq = [[self.pos_to_tag(p) for p in pos] for pos in pos_seq]
        self.pos_seq = [[[pos_seq[index][i] for _ in range(len(words_seq[index][i]))]
                         for i in range(len(pos_seq[index]))] for index in range(len(pos_seq))]
        self.tag_seq = [[[self.tag_perform(tag_seq[index][i], w) for w in range(len(words_seq[index][i]))]
                         for i in range(len(tag_seq[index]))] for index in range(len(tag_seq))]
        self.pos_seq = [[u'un'] + [self.pos_to_tag(p) for pos in pos_seq for p in pos] + [u'un'] for pos_seq in self.pos_seq]
        self.tag_seq = [[t for tag in tag_seq for t in tag] for tag_seq in self.tag_seq]
        self.word_seq = [[u'<BOS>'] + [w for word in word_seq for w in word] + [u'<EOS>'] for word_seq in words_seq]

    # 特征选取
    def extract_feature(self, word_grams):
        features, feature_list = [], []
        for index in range(len(word_grams)):
            for i in range(len(word_grams[index])):
                word_gram = word_grams[index][i]
                feature = {u'w-1': word_gram[0],
                           u'w': word_gram[1],
                           u'w+1': word_gram[2],
                           u'w-1:w': word_gram[0] + word_gram[1],
                           u'w:w+1': word_gram[1] + word_gram[2],
                           u'bias': 1.0}
                feature_list.append(feature)
            features.append(feature_list)
            feature_list = []
        return features

    # 窗口切分
    def segment_by_window(self, words_list=None, window=3):
        words = []
        begin, end = 0, window
        for _ in range(1, len(words_list)):
            if end > len(words_list): break
            words.append(words_list[begin:end])
            begin = begin + 1
            end = end + 1
        return words

    # 训练数据
    def generator(self):
        word_grams = [self.segment_by_window(word_list) for word_list in self.word_seq]
        features = self.extract_feature(word_grams)
        return features, self.tag_seq


# 定义一个类来使用CRF模型进行命名实体识别
class CRF_NER(object):
    def __init__(self):
        # 初始化参数
        self.algorithm = "lbfgs"  # CRF算法
        self.c1 = "0.1"
        self.c2 = "0.1"
        self.max_iterations = 100  # 最大迭代次数
        self.model_path = "output/sklearn_crf_model.pkl"
        self.corpus = CorpusProcess()  # 创建CorpusProcess实例
        self.corpus.initialize()  # 初始化语料
        self.model = None  # 初始化模型为None

    # 初始化模型
    def initialize_model(self):
        algorithm = self.algorithm
        c1 = float(self.c1)
        c2 = float(self.c2)
        max_iterations = int(self.max_iterations)
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2,
                                          max_iterations=max_iterations,
                                          all_possible_transitions=True)

    # 训练模型
    def train(self):
        self.initialize_model()
        x, y = self.corpus.generator()
        train_size = int(len(x) * 0.8)
        # 划分数据集为训练集和测试集
        x_train, y_train = x[:train_size], y[:train_size]
        x_test, y_test = x[train_size:], y[train_size:]
        self.model.fit(x_train, y_train)
        labels = list(self.model.classes_)
        labels.remove('O')  # 移除O标签
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        # 转换为二进制表示用于计算精确率、召回率和F1值
        mlb = MultiLabelBinarizer()
        y_test_binary = mlb.fit_transform([['O' if t == 'O' else t[2:] for t in tags] for tags in y_test])
        y_predict_binary = mlb.transform([['O' if t == 'O' else t[2:] for t in tags] for tags in y_predict])
        # 计算精确率、召回率和F1值
        precision = precision_score(y_test_binary, y_predict_binary, average='weighted')
        recall = recall_score(y_test_binary, y_predict_binary, average='weighted')
        f1 = f1_score(y_test_binary, y_predict_binary, average='weighted')
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
        self.save_model()

    # 保存模型
    def save_model(self):
        joblib.dump(self.model, self.model_path)

    # 加载模型
    def load_model(self):
        self.model = joblib.load(self.model_path)

    def predict(self, sentence):
        self.load_model()
        u_sent = sentence
        word_lists = [[u'<BOS>'] + [c for c in u_sent] + [u'<EOS>']]
        word_grams = [self.corpus.segment_by_window(word_list) for word_list in word_lists]
        features = self.corpus.extract_feature(word_grams)
        y_predict = self.model.predict(features)
        entities = []
        start = -1
        for index in range(len(y_predict[0])):
            if y_predict[0][index] != u'O':
                if start == -1:
                    start = index
            else:
                if start != -1:
                    entities.append(u_sent[start:index])
                    start = -1
        # 添加最后一个实体
        if start != -1:
            entities.append(u_sent[start:])
        return entities


# 创建CRF_NER实例并训练模型
ner = CRF_NER()
ner.train()

sentence = """
在广袤的国土之上，多彩纷呈的各地佳肴如同丰富的画卷，呈现出各自的独特魅力。
在广东潮汕，人们喜爱吃牛肉火锅，将新鲜的牛肉在滚烫的汤底中涮煮，美味可口。
在山东德州，人们喜欢品尝那香气扑鼻的扒鸡，外酥里嫩。
在江苏苏州，人们喜爱食用酱鸭，其独特的口味和制作工艺深受人们喜爱，每一口都让人回味无穷。
在浙江杭州，人们喜爱品尝那肥而不腻、入口即化的东坡肉，以及那清淡可口的西湖醋鱼，感受那江南水乡的韵味。 
食物文化，是中华民族五千年文明的瑰宝，它不仅满足了人们的味蕾，更传承着中华民族的文化精髓。
"""

# 预测实体
output = ner.predict(sentence)
print("识别到的实体：", output)
