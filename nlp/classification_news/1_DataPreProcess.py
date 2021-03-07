import numpy as np

# 字典
# 文章字典{'文章类型':出现次数}
ArticleDic = {}
# 类型字典{'文章类型':{词语id:词频}}
ClassDic = {}
# 词语字典{词语id:词频}
WordDic = {}
# 文章词语字典{'文章类型':总词数}
ClassWordCount = {}

# 概率
# 文章先验概率{'文章类型':先验概率}
ClassicProb = {}
# 词语先验概率{'文章类型':{'词语id':先验概率}}
ClassWordProb = {}
# 初始频率，未出现过的词的词频
DefaultFreq = 0.1
# 初始概率，未出现过的词的概率{'文章类型':默认先验概率}
DefaultPriorProb = {}

# 训练数据
file = open("./simple_train.csv", "r")


# 读取训练数据
def load_data():
    for line in file.readlines():
        # 删除第一行数据
        if len(line) > 20:
            label_article = line.strip().split("\t")

            # step1.读取文章，存储格式{'文章类型':出现次数}
            label = label_article[0]
            if label not in ArticleDic:
                ArticleDic[label] = 1
                ClassDic[label] = {}
                ClassWordCount[label] = 0
            else:
                ArticleDic[label] += 1

            # step2 获取每篇文章的词语，存储格式{'文章类型':{词语id:词频}}
            words = label_article[1].strip().split(" ")
            for word in words:
                if word not in ClassDic[label]:
                    ClassDic[label][word] = 1
                else:
                    ClassDic[label][word] += 1

                # step3 生成词语字典，存储格式{词语id:词频}
                if word not in WordDic:
                    WordDic[word] = 1
                else:
                    WordDic[word] += 1

            ClassWordCount[label] = len(words) + ClassWordCount[label]

    np.save("FileArticleDic.npy", ArticleDic)
    np.save("FileClassDic.npy", ClassDic)
    np.save("FileWordDic.npy", WordDic)
    np.save("FileClassWordCount.npy", ClassWordCount)


def calculate_model():
    # 文章总数
    articleSum = 0
    # 词语总数
    wordSum = 0

    for word in WordDic:
        wordSum += WordDic[word]

    for label in ArticleDic:
        articleSum += ArticleDic[label]

    # step1 各类文章先验概率
    for label in ArticleDic:
        ClassicProb[label] = ArticleDic[label] / articleSum

    # step2 各类文章中每个词语的先验概率
    for label in ClassDic:
        if label not in ClassWordProb:
            ClassWordProb[label] = {}

        for word in ClassDic[label]:
            ClassWordProb[label][word] = ClassDic[label][word] / ClassWordCount[label]

        # step3 各类文章未出现词汇的先验概率
        DefaultPriorProb[label] = DefaultFreq / ClassWordCount[label]

    np.save("FileClassWordProb.npy", ClassWordProb)
    np.save("FileClassicProb.npy", ClassicProb)
    np.save("FileDefaultPriorProb.npy", DefaultPriorProb)


load_data()
calculate_model()
