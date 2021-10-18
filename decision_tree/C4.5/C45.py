# -*- coding: cp936 -*-
from math import log
import operator
import os

import re
from numpy import inf
import copy


# 计算信息熵
def calcShannonEnt(dataSet, labelIndex):
    # type: (list) -> float
    numEntries = 0  # 样本数(按权重计算）
    labelCounts = {}
    for featVec in dataSet:  # 遍历每个样本
        if featVec[labelIndex] != 'N':
            weight = float(featVec[-2])
            numEntries += weight
            currentLabel = featVec[-1]  # 当前样本的类别
            if currentLabel not in labelCounts.keys():  # 生成类别字典
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += weight  # 数据集的倒数第二个值用来标记样本权重
    shannonEnt = 0.0
    for key in labelCounts:  # 计算信息熵
        prob = float(labelCounts[key]) / numEntries
        shannonEnt = shannonEnt - prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value, LorR='N'):
    """
    type: (list, int, string or float, string) -> list
    划分数据集
    axis:按第几个特征划分
    value:划分特征的值
    LorR: N 离散属性; L 小于等于value值; R 大于value值
    """
    retDataSet = []
    featVec = []
    if LorR == 'N':  # 离散属性
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
    elif LorR == 'L':
        for featVec in dataSet:
            if featVec[axis] != 'N':
                if float(featVec[axis]) < value:
                    retDataSet.append(featVec)
    elif LorR == 'R':
        for featVec in dataSet:
            if featVec[axis] != 'N':
                if float(featVec[axis]) > value:
                    retDataSet.append(featVec)
    return retDataSet


def splitDataSetWithNull(dataSet, axis, value, LorR='N'):
    """
    type: (list, int, string or float, string) -> list
    划分数据集
    axis:按第几个特征划分
    value:划分特征的值
    LorR: N 离散属性; L 小于等于value值; R 大于value值
    """
    retDataSet = []
    nullDataSet = []
    featVec = []
    totalWeightV = calcTotalWeight(dataSet, axis, False)  # 非空样本权重
    totalWeightSub = 0.0
    if LorR == 'N':  # 离散属性
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
            elif featVec[axis] == 'N':
                reducedNullVec = featVec[:axis]
                reducedNullVec.extend(featVec[axis + 1:])
                nullDataSet.append(reducedNullVec)
    elif LorR == 'L':
        for featVec in dataSet:
            if featVec[axis] != 'N':
                if float(featVec[axis]) < value:
                    retDataSet.append(featVec)
            elif featVec[axis] == 'N':
                nullDataSet.append(featVec)
    elif LorR == 'R':
        for featVec in dataSet:
            if featVec[axis] != 'N':
                if float(featVec[axis]) > value:
                    retDataSet.append(featVec)
            elif featVec[axis] == 'N':
                nullDataSet.append(featVec)

    totalWeightSub = calcTotalWeight(retDataSet, -1, True)  # 计算此分支中非空样本的总权重
    for nullVec in nullDataSet:  # 把缺失值样本按权值比例划分到分支中
        nullVec[-2] = float(nullVec[-2]) * totalWeightSub / totalWeightV
        retDataSet.append(nullVec)

    return retDataSet


def calcTotalWeight(dataSet, labelIndex, isContainNull):
    """
    type: (list, int, bool) -> float
    计算样本集对某个特征值的总样本树（按权重计算）
    :param dataSet: 数据集
    :param labelIndex: 特征值索引
    :param isContainNull: 是否包含空值的样本
    :return: 返回样本集的总权重值
    """
    totalWeight = 0.0
    print(len(dataSet))
    for featVec in dataSet:  # 遍历每个样本
        print(featVec)
        weight = float(featVec[-2])
        if isContainNull is False and featVec[labelIndex] != 'N':
            totalWeight += weight  # 非空样本树，按权重计算
        if isContainNull is True:
            totalWeight += weight  # 总样本数，按权重计算
    return totalWeight


def calcGain(dataSet, labelIndex, labelPropertyi):
    """
    type: (list, int, int) -> float, int
    计算信息增益,返回信息增益值和连续属性的划分点
    dataSet: 数据集
    labelIndex: 特征值索引
    labelPropertyi: 特征值类型，0为离散，1为连续
    """
    baseEntropy = calcShannonEnt(dataSet, labelIndex)  # 计算根节点的信息熵
    featList = [example[labelIndex] for example in dataSet]  # 特征值列表
    uniqueVals = set(featList)  # 该特征包含的所有值
    newEntropy = 0.0
    totalWeight = 0.0
    totalWeightV = 0.0
    totalWeight = calcTotalWeight(dataSet, labelIndex, True)  # 总样本权重
    totalWeightV = calcTotalWeight(dataSet, labelIndex, False)  # 非空样本权重
    if labelPropertyi == 0:  # 对离散的特征
        for value in uniqueVals:  # 对每个特征值，划分数据集, 计算各子集的信息熵
            if value != 'N':
                subDataSet = splitDataSet(dataSet, labelIndex, value)
                totalWeightSub = 0.0
                totalWeightSub = calcTotalWeight(subDataSet, labelIndex, True)
                prob = totalWeightSub / totalWeightV
                newEntropy += prob * calcShannonEnt(subDataSet, labelIndex)
    else:  # 对连续的特征
        uniqueValsList = list(uniqueVals)
        if 'N' in uniqueValsList:
            uniqueValsList.remove('N')
        sortedUniqueVals = sorted(uniqueValsList)  # 对特征值排序
        listPartition = []
        minEntropy = inf
        if len(sortedUniqueVals) == 1:  # 如果只有一个值，可以看作只有左子集，没有右子集
            totalWeightLeft = calcTotalWeight(dataSet, labelIndex, True)
            probLeft = totalWeightLeft / totalWeightV
            minEntropy = probLeft * calcShannonEnt(dataSet, labelIndex)
        else:
            for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
                partValue = (float(sortedUniqueVals[j]) + float(
                    sortedUniqueVals[j + 1])) / 2
                # 对每个划分点，计算信息熵
                dataSetLeft = splitDataSet(dataSet, labelIndex, partValue, 'L')
                dataSetRight = splitDataSet(dataSet, labelIndex, partValue, 'R')
                totalWeightLeft = 0.0
                totalWeightLeft = calcTotalWeight(dataSetLeft, labelIndex, True)
                totalWeightRight = 0.0
                totalWeightRight = calcTotalWeight(dataSetRight, labelIndex, True)
                probLeft = totalWeightLeft / totalWeightV
                probRight = totalWeightRight / totalWeightV
                Entropy = probLeft * calcShannonEnt(dataSetLeft, labelIndex) + \
                          probRight * calcShannonEnt(dataSetRight, labelIndex)
                if Entropy < minEntropy:  # 取最小的信息熵
                    minEntropy = Entropy
        newEntropy = minEntropy
    gain = totalWeightV / totalWeight * (baseEntropy - newEntropy)
    return gain


def calcGainRatio(dataSet, labelIndex, labelPropertyi):
    """
    type: (list, int, int) -> float, int
    计算信息增益率,返回信息增益率和连续属性的划分点
    dataSet: 数据集
    labelIndex: 特征值索引
    labelPropertyi: 特征值类型，0为离散，1为连续
    """
    baseEntropy = calcShannonEnt(dataSet, labelIndex)  # 计算根节点的信息熵
    featList = [example[labelIndex] for example in dataSet]  # 特征值列表
    uniqueVals = set(featList)  # 该特征包含的所有值
    newEntropy = 0.0
    bestPartValuei = None
    IV = 0.0
    totalWeight = 0.0
    totalWeightV = 0.0
    totalWeight = calcTotalWeight(dataSet, labelIndex, True)  # 总样本权重
    totalWeightV = calcTotalWeight(dataSet, labelIndex, False)  # 非空样本权重
    if labelPropertyi == 0:  # 对离散的特征
        for value in uniqueVals:  # 对每个特征值，划分数据集, 计算各子集的信息熵
            subDataSet = splitDataSet(dataSet, labelIndex, value)
            totalWeightSub = 0.0
            totalWeightSub = calcTotalWeight(subDataSet, labelIndex, True)
            if value != 'N':
                prob = totalWeightSub / totalWeightV
                newEntropy += prob * calcShannonEnt(subDataSet, labelIndex)
            prob1 = totalWeightSub / totalWeight
            IV -= prob1 * log(prob1, 2)
    else:  # 对连续的特征
        uniqueValsList = list(uniqueVals)
        if 'N' in uniqueValsList:
            uniqueValsList.remove('N')
            # 计算空值样本的总权重，用于计算IV
            totalWeightN = 0.0
            dataSetNull = splitDataSet(dataSet, labelIndex, 'N')
            totalWeightN = calcTotalWeight(dataSetNull, labelIndex, True)
            probNull = totalWeightN / totalWeight
            if probNull > 0.0:
                IV += -1 * probNull * log(probNull, 2)

        sortedUniqueVals = sorted(uniqueValsList)  # 对特征值排序
        listPartition = []
        minEntropy = inf

        if len(sortedUniqueVals) == 1:  # 如果只有一个值，可以看作只有左子集，没有右子集
            totalWeightLeft = calcTotalWeight(dataSet, labelIndex, True)
            probLeft = totalWeightLeft / totalWeightV
            minEntropy = probLeft * calcShannonEnt(dataSet, labelIndex)
            IV = -1 * probLeft * log(probLeft, 2)
        else:
            for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
                partValue = (float(sortedUniqueVals[j]) + float(
                    sortedUniqueVals[j + 1])) / 2
                # 对每个划分点，计算信息熵
                dataSetLeft = splitDataSet(dataSet, labelIndex, partValue, 'L')
                dataSetRight = splitDataSet(dataSet, labelIndex, partValue, 'R')
                totalWeightLeft = 0.0
                totalWeightLeft = calcTotalWeight(dataSetLeft, labelIndex, True)
                totalWeightRight = 0.0
                totalWeightRight = calcTotalWeight(dataSetRight, labelIndex, True)
                probLeft = totalWeightLeft / totalWeightV
                probRight = totalWeightRight / totalWeightV
                Entropy = probLeft * calcShannonEnt(
                    dataSetLeft, labelIndex) + probRight * calcShannonEnt(dataSetRight, labelIndex)
                if Entropy < minEntropy:  # 取最小的信息熵
                    minEntropy = Entropy
                    bestPartValuei = partValue
                    probLeft1 = totalWeightLeft / totalWeight
                    probRight1 = totalWeightRight / totalWeight
                    IV += -1 * (probLeft1 * log(probLeft1, 2) + probRight1 * log(probRight1, 2))

        newEntropy = minEntropy
    gain = totalWeightV / totalWeight * (baseEntropy - newEntropy)
    if IV == 0.0:  # 如果属性只有一个值，IV为0，为避免除数为0，给个很小的值
        IV = 0.0000000001
    gainRatio = gain / IV
    return gainRatio, bestPartValuei


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet, labelProperty):
    """
    type: (list, int) -> int, float
    :param dataSet: 样本集
    :param labelProperty: 特征值类型，1 连续， 0 离散
    :return: 最佳划分属性的索引和连续属性的划分值
    """
    numFeatures = len(labelProperty)  # 特征数
    bestInfoGainRatio = 0.0
    bestFeature = -1
    bestPartValue = None  # 连续的特征值，最佳划分值
    gainSum = 0.0
    gainAvg = 0.0
    for i in range(numFeatures):  # 对每个特征循环
        infoGain = calcGain(dataSet, i, labelProperty[i])
        gainSum += infoGain
    gainAvg = gainSum / numFeatures
    for i in range(numFeatures):  # 对每个特征循环
        infoGainRatio, bestPartValuei = calcGainRatio(dataSet, i, labelProperty[i])
        infoGain = calcGain(dataSet, i, labelProperty[i])
        if infoGainRatio > bestInfoGainRatio and infoGain > gainAvg:  # 取信息增益高于平均增益且信息增益率最大的特征
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
            bestPartValue = bestPartValuei
    return bestFeature, bestPartValue


# 通过排序返回出现次数最多的类别
def majorityCnt(classList, weightList):
    classCount = {}
    for i in range(len(classList)):
        if classList[i] not in classCount.keys():
            classCount[classList[i]] = 0.0
        classCount[classList[i]] += round(float(weightList[i]),1)

    # python 2.7
    # sortedClassCount = sorted(classCount.iteritems(),
    #                         key=operator.itemgetter(1), reverse=True)
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    if len(sortedClassCount) == 1:
        return (sortedClassCount[0][0],sortedClassCount[0][1],0.0)
    return (sortedClassCount[0][0], sortedClassCount[0][1], sortedClassCount[1][1])


# 创建树, 样本集 特征 特征属性（0 离散， 1 连续）
def createTree(dataSet, labels, labelProperty):
    classList = [example[-1] for example in dataSet]  # 类别向量
    weightList = [example[-2] for example in dataSet]  # 权重向量
    if classList.count(classList[0]) == len(classList):  # 如果只有一个类别，返回
        totalWeiht = calcTotalWeight(dataSet,0,True)
        return (classList[0], round(totalWeiht,1),0.0)
    #totalWeight = calcTotalWeight(dataSet, 0, True)
    if len(dataSet[0]) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat, bestPartValue = chooseBestFeatureToSplit(dataSet,
                                                       labelProperty)  # 最优分类特征的索引
    if bestFeat == -1:  # 如果无法选出最优分类特征，返回出现次数最多的类别
        return majorityCnt(classList, weightList)
    if labelProperty[bestFeat] == 0:  # 对离散的特征
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        labelsNew = copy.copy(labels)
        labelPropertyNew = copy.copy(labelProperty)
        del (labelsNew[bestFeat])  # 已经选择的特征不再参与分类
        del (labelPropertyNew[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueValue = set(featValues)  # 该特征包含的所有值
        uniqueValue.discard('N')
        for value in uniqueValue:  # 对每个特征值，递归构建树
            subLabels = labelsNew[:]
            subLabelProperty = labelPropertyNew[:]
            myTree[bestFeatLabel][value] = createTree(
                splitDataSetWithNull(dataSet, bestFeat, value), subLabels,
                subLabelProperty)
    else:  # 对连续的特征，不删除该特征，分别构建左子树和右子树
        bestFeatLabel = labels[bestFeat] + '<' + str(bestPartValue)
        myTree = {bestFeatLabel: {}}
        subLabels = labels[:]
        subLabelProperty = labelProperty[:]
        # 构建左子树
        valueLeft = 'Y'
        myTree[bestFeatLabel][valueLeft] = createTree(
            splitDataSetWithNull(dataSet, bestFeat, bestPartValue, 'L'), subLabels,
            subLabelProperty)
        # 构建右子树
        valueRight = 'N'
        myTree[bestFeatLabel][valueRight] = createTree(
            splitDataSetWithNull(dataSet, bestFeat, bestPartValue, 'R'), subLabels,
            subLabelProperty)
    return myTree


# 测试算法
def classify(inputTree, classList, featLabels, featLabelProperties, testVec):
    firstStr = list(inputTree.keys())[0]  # 根节点
    firstLabel = firstStr
    lessIndex = str(firstStr).find('<')
    if lessIndex > -1:  # 如果是连续型的特征
        firstLabel = str(firstStr)[:lessIndex]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstLabel)  # 跟节点对应的特征
    classLabel = {}
    for classI in classList:
        classLabel[classI] = 0.0
    for key in secondDict.keys():  # 对每个分支循环
        if featLabelProperties[featIndex] == 0:  # 离散的特征
            if testVec[featIndex] == key:  # 测试样本进入某个分支
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabelSub = classify(secondDict[key], classList, featLabels,
                                          featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[classKey]
                else:  # 如果是叶子， 返回结果
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict[key][1]
                        else:
                            classLabel[classKey] += secondDict[key][2]
            elif testVec[featIndex] == 'N':  # 如果测试样本的属性值缺失，则进入每个分支
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabelSub = classify(secondDict[key], classList, featLabels,
                                          featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[key]
                else:  # 如果是叶子， 返回结果
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict[key][1]
                        else:
                            classLabel[classKey] += secondDict[key][2]
        else:
            partValue = float(str(firstStr)[lessIndex + 1:])
            if testVec[featIndex] == 'N':  # 如果测试样本的属性值缺失，则对每个分支的结果加和
                # 进入左子树
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabelSub = classify(secondDict[key], classList, featLabels,
                                          featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[classKey]
                else:  # 如果是叶子， 返回结果
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict[key][1]
                        else:
                            classLabel[classKey] += secondDict[key][2]
            elif float(testVec[featIndex]) <= partValue and key == 'Y':  # 进入左子树
                if type(secondDict['Y']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabelSub = classify(secondDict['Y'], classList, featLabels,
                                             featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[classKey]
                else:  # 如果是叶子， 返回结果
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict['Y'][1]
                        else:
                            classLabel[classKey] += secondDict['Y'][2]
            elif float(testVec[featIndex]) > partValue and key == 'N':
                if type(secondDict['N']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabelSub = classify(secondDict['N'], classList, featLabels,
                                             featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[classKey]
                else:  # 如果是叶子， 返回结果
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict['N'][1]
                        else:
                            classLabel[classKey] += secondDict['N'][2]

    return classLabel


# 存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


# 读取决策树, 文件不存在返回None
def grabTree(filename):
    import pickle
    if os.path.isfile(filename):
        fr = open(filename)
        return pickle.load(fr)
    else:
        return None


# 测试决策树正确率
def testing(myTree, classList, data_test, labels, labelProperties):
    error = 0.0
    for i in range(len(data_test)):
        classLabelSet = classify(myTree, classList, labels, labelProperties, data_test[i])
        maxWeight = 0.0
        classLabel = ''
        for item in classLabelSet.items():
            if item[1] > maxWeight:
                classLabel = item[0]
        if classLabel !=  data_test[i][-1]:
            error += 1
    return float(error)


# 测试投票节点正确率
def testingMajor(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major[0] != data_test[i][-1]:
            error += 1
    # print 'major %d' %error
    return float(error)


# 后剪枝
def postPruningTree(inputTree, classSet, dataSet, data_test, labels, labelProperties):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    classList = [example[-1] for example in dataSet]
    weightList = [example[-2] for example in dataSet]
    featkey = copy.deepcopy(firstStr)
    if '<' in firstStr:  # 对连续的特征值，使用正则表达式获得特征标签和value
        featkey = re.compile("(.+<)").search(firstStr).group()[:-1]
        featvalue = float(re.compile("(<.+)").search(firstStr).group()[1:])
    labelIndex = labels.index(featkey)
    temp_labels = copy.deepcopy(labels)
    temp_labelProperties = copy.deepcopy(labelProperties)
    if labelProperties[labelIndex] == 0:  # 离散特征
        del (labels[labelIndex])
        del (labelProperties[labelIndex])
    for key in secondDict.keys():  # 对每个分支
        if type(secondDict[key]).__name__ == 'dict':  # 如果不是叶子节点
            if temp_labelProperties[labelIndex] == 0:  # 离散的
                subDataSet = splitDataSet(dataSet, labelIndex, key)
                subDataTest = splitDataSet(data_test, labelIndex, key)
            else:
                if key == 'Y':
                    subDataSet = splitDataSet(dataSet, labelIndex, featvalue,
                                              'L')
                    subDataTest = splitDataSet(data_test, labelIndex,
                                               featvalue, 'L')
                else:
                    subDataSet = splitDataSet(dataSet, labelIndex, featvalue,
                                              'R')
                    subDataTest = splitDataSet(data_test, labelIndex,
                                               featvalue, 'R')
            if len(subDataTest) > 0:
                inputTree[firstStr][key] = postPruningTree(secondDict[key], classSet,
                                                       subDataSet, subDataTest,
                                                       copy.deepcopy(labels),
                                                       copy.deepcopy(
                                                           labelProperties))
    if testing(inputTree, classSet, data_test, temp_labels,
               temp_labelProperties) <= testingMajor(majorityCnt(classList, weightList),
                                                     data_test):
        return inputTree
    return majorityCnt(classList,weightList)
