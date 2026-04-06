# 代码4-3 双向最大匹配法分词

dictionary = ['北京市', '北京市民', '民办高中', '天安门广场', '高中']
text = '北京市民办高中'


def forward_maximum_matching(text, dictionary):
    word_segmentation = []  # 用于存储分词结果
    text_length = len(text)  # 待分词文本长度
    while text_length > 0:
        # 限制匹配词语的最大长度为字典长度或剩余文本长度，取两者中较小的值
        max_word_length = min(text_length, len(dictionary))
        # 正向最大匹配
        forward_word = text[:max_word_length]  # 从文本开头位置截取最大长度的文本进行匹配
        while forward_word not in dictionary:  # 若当前截取文本不在词典中
            # 若截取文本长度为1，则无法再进行切分，将其作为一个独立词语
            if len(forward_word) == 1:
                break
            forward_word = forward_word[:-1]  # 若不是独立词语，则减少截取长度继续尝试匹配
        word_segmentation.append(forward_word)  # 将匹配到的词语添加到分词结果
        text = text[len(forward_word):]  # 更新待分词文本（去除已匹配的部分）
        text_length = len(text)  # 更新待分词文本长度
    return word_segmentation


# 定义逆向最大匹配函数
def backward_maximum_matching(text, dictionary):
    result = []  # 存储切分结果
    max_len = max(len(word) for word in dictionary)
    while len(text) > 0:
        word = None
        for i in range(max_len, 0, -1):  # 从最大长度开始尝试匹配
            if text[-i:] in dictionary:
                word = text[-i:]  # 匹配到了词语
                break
        if word is None:  # 如果没有匹配到词语，则默认切分一个字
            word = text[-1:]
        result.insert(0, word)  # 将切分结果插入到列表的开头
        text = text[:-len(word)]  # 更新待切分的文本
    return result


# 定义双向最大匹配函数
def bidirectional_maximum_matching(text, dictionary):
    # 获取正向最大匹配结果
    forward_result = forward_maximum_matching(text, dictionary)
    # 获取逆向最大匹配结果
    backward_result = backward_maximum_matching(text, dictionary)
    if len(forward_result) < len(backward_result):
        return forward_result
    elif len(forward_result) > len(backward_result):
        return backward_result
    else:
        # 如果分词数量相同，则选择单字词较少的结果
        forward_single_word_count = sum(1 for word in forward_result if len(word) == 1)
        backward_single_word_count = sum(1 for word in backward_result if len(word) == 1)
        if forward_single_word_count < backward_single_word_count:
            return forward_result
        else:
            return backward_result


# 测试双向最大匹配函数
bidirectional_result = bidirectional_maximum_matching(text, dictionary)
print(bidirectional_result)
