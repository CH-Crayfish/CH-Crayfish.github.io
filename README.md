统计学习方法笔记（自用）
====   
朴素贝叶斯
-
除了MultinomialNB(多项式分布的朴素贝叶斯)之外，还有GaussianNB就是先验为高斯分布的朴素贝叶斯，BernoulliNB就是先验为伯努利分布的朴素贝叶斯
```
import numpy as np
from functools import reduce
from sklearn.naive_bayes import MultinomialNB

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak','how','to', 'stop','him'],
                  ['quit','buying','worthless','dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0,1] #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec # 返回实验样本切分的词条和类别标签向量
    def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空的不重复列表（为什么列表要用小括号）
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

'''
函数功能：根据vocabList词汇表，将inputSet向量化，向量的每一个元素为1或0
参数说明：
    vocabList:词汇表
    inputSet:切分好的词条列表中的一条
    返回：
    returnVec:文档向量，词集模型
'''
def set0fWord2Vec(vocabList, inputSet): #将单词转化为向量，便于进行计算
    returnVec = [0] * len(vocabList) #创建一个其中所含元素都为0的向量
    for word in inputSet: #遍历每个词条
        if word in inputSet: #如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print('词汇： %s 并没有在词汇表中' %word)#词汇表中没有这个单词，表示出现了问题
    return returnVec #返回文档向量
 def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) #训练集中样本数量
    numWords = len(trainMatrix[0])  #每一条样本的词条数量
    pAbusive = sum(trainCategory)/float(numTrainDocs) #文档属于侮辱类的概率
    p0Num = np.ones(numWords);p1Num = np.ones(numWords) #词条初始化次数为1，避免出现0的情况。拉普拉斯平滑第一步
    p0Denom = 2.0; p1Denom = 2.0 #分母初始化为2.0，拉普拉斯平滑第而二步
    for i in range (numTrainDocs):#对每个标签进行判断，6次
        if trainCategory[i] == 1: #统计属于侮辱类的条件概率所需的数据，即P(w0/1),P(w1/1),P(w2/1)...
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else: #统计属于非侮辱类的条件概率所需的数据，即P（w0/0）,P(w1/0),P(w2/0)...
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            
    p1Vect = np.log(p1Num/p1Denom)  #相除，然后取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)  
    return p0Vect, p1Vect,pAbusive #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)#对应元素相乘，log(A*B)=logA + logB
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1>p0:
        return 1
    else:
        return 0
 def testingNB():
    list0Posts, listClasses = loadDataSet()#创建实验样本
    myVocabList = createVocabList(list0Posts)#创建词汇表
    trainMat=[]
    for postinDoc in list0Posts:
        trainMat.append(set0fWord2Vec(myVocabList, postinDoc))#将实验样本向量化
    p0V, p1V,pAb = trainNBO(np.array(trainMat), np.array(listClasses))#训练朴素贝叶斯分类器
    
    testEntry = ['love', 'my', 'dalmation']#测试样本1
    thisDoc = np.array(set0fWord2Vec(myVocabList, testEntry))#测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')#执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')#执行分类并打印分类结果
    testEntry = ['stupid','garbage']#测试样本2
    
    thisDoc = np.array(set0fWord2Vec(myVocabList, testEntry))#测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')#执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')#执行分类并打印分类结果
        
if __name__ == '__main__':
    testingNB()
