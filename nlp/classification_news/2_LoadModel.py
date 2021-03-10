import numpy as np
import math


# 概率
# 文章先验概率{'文章类型':先验概率}
ClassicProb = {}
# 词语先验概率{'文章类型':{'词语id':先验概率}}
ClassWordProb = {}
# 初始概率，未出现过的词的概率{'文章类型':默认先验概率}
DefaultPriorProb = {}
# 结果
ResultList = []

# 截取的部分数据
# file = open("./simple_test.csv", "r")
# ResultFile = "./simple_res.csv"
file = open("./test_a.csv", "r")
ResultFile = "./test_res.csv"


def load_model():
    global ClassWordProb
    global ClassicProb
    global DefaultPriorProb

    ClassWordProb = np.load("FileClassWordProb.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').item()
    ClassicProb = np.load("FileClassicProb.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').item()
    DefaultPriorProb = np.load("FileDefaultPriorProb.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').item()


def evaluate():
    for line in file.readlines():
        if len(line) > 20:
            words = line.strip().split(" ")

            # 测试分数{'文章类型':预测分数}
            score_dic = {}
            for label in ClassWordProb.keys():
                score_dic[label] = 0

                for word in words:
                    if word in ClassWordProb[label]:
                        score_dic[label] += math.log(ClassWordProb[label][word])
                    else:
                        score_dic[label] += math.log(DefaultPriorProb[label])

            max_predict_score = max(score_dic.values())
            for label in score_dic.keys():
                if score_dic[label] == max_predict_score:
                    ResultList.append(label)


def output_result():
    i = 0
    outfile = open(ResultFile, 'w')
    outfile.write("label")
    outfile.write('\n')
    while i < len(ResultList):
        outfile.write(str(ResultList[i]))
        outfile.write('\n')
        i += 1
    outfile.close()


load_model()
evaluate()
output_result()
