# -*- coding: cp936 -*-
from math import log
import operator
import os

import re
from numpy import inf
import copy


# ������Ϣ��
def calcShannonEnt(dataSet, labelIndex):
    # type: (list) -> float
    numEntries = 0  # ������(��Ȩ�ؼ��㣩
    labelCounts = {}
    for featVec in dataSet:  # ����ÿ������
        if featVec[labelIndex] != 'N':
            weight = float(featVec[-2])
            numEntries += weight
            currentLabel = featVec[-1]  # ��ǰ���������
            if currentLabel not in labelCounts.keys():  # ��������ֵ�
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += weight  # ���ݼ��ĵ����ڶ���ֵ�����������Ȩ��
    shannonEnt = 0.0
    for key in labelCounts:  # ������Ϣ��
        prob = float(labelCounts[key]) / numEntries
        shannonEnt = shannonEnt - prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value, LorR='N'):
    """
    type: (list, int, string or float, string) -> list
    �������ݼ�
    axis:���ڼ�����������
    value:����������ֵ
    LorR: N ��ɢ����; L С�ڵ���valueֵ; R ����valueֵ
    """
    retDataSet = []
    featVec = []
    if LorR == 'N':  # ��ɢ����
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
    �������ݼ�
    axis:���ڼ�����������
    value:����������ֵ
    LorR: N ��ɢ����; L С�ڵ���valueֵ; R ����valueֵ
    """
    retDataSet = []
    nullDataSet = []
    featVec = []
    totalWeightV = calcTotalWeight(dataSet, axis, False)  # �ǿ�����Ȩ��
    totalWeightSub = 0.0
    if LorR == 'N':  # ��ɢ����
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

    totalWeightSub = calcTotalWeight(retDataSet, -1, True)  # ����˷�֧�зǿ���������Ȩ��
    for nullVec in nullDataSet:  # ��ȱʧֵ������Ȩֵ�������ֵ���֧��
        nullVec[-2] = float(nullVec[-2]) * totalWeightSub / totalWeightV
        retDataSet.append(nullVec)

    return retDataSet


def calcTotalWeight(dataSet, labelIndex, isContainNull):
    """
    type: (list, int, bool) -> float
    ������������ĳ������ֵ��������������Ȩ�ؼ��㣩
    :param dataSet: ���ݼ�
    :param labelIndex: ����ֵ����
    :param isContainNull: �Ƿ������ֵ������
    :return: ��������������Ȩ��ֵ
    """
    totalWeight = 0.0
    print(len(dataSet))
    for featVec in dataSet:  # ����ÿ������
        print(featVec)
        weight = float(featVec[-2])
        if isContainNull is False and featVec[labelIndex] != 'N':
            totalWeight += weight  # �ǿ�����������Ȩ�ؼ���
        if isContainNull is True:
            totalWeight += weight  # ������������Ȩ�ؼ���
    return totalWeight


def calcGain(dataSet, labelIndex, labelPropertyi):
    """
    type: (list, int, int) -> float, int
    ������Ϣ����,������Ϣ����ֵ���������ԵĻ��ֵ�
    dataSet: ���ݼ�
    labelIndex: ����ֵ����
    labelPropertyi: ����ֵ���ͣ�0Ϊ��ɢ��1Ϊ����
    """
    baseEntropy = calcShannonEnt(dataSet, labelIndex)  # ������ڵ����Ϣ��
    featList = [example[labelIndex] for example in dataSet]  # ����ֵ�б�
    uniqueVals = set(featList)  # ����������������ֵ
    newEntropy = 0.0
    totalWeight = 0.0
    totalWeightV = 0.0
    totalWeight = calcTotalWeight(dataSet, labelIndex, True)  # ������Ȩ��
    totalWeightV = calcTotalWeight(dataSet, labelIndex, False)  # �ǿ�����Ȩ��
    if labelPropertyi == 0:  # ����ɢ������
        for value in uniqueVals:  # ��ÿ������ֵ���������ݼ�, ������Ӽ�����Ϣ��
            if value != 'N':
                subDataSet = splitDataSet(dataSet, labelIndex, value)
                totalWeightSub = 0.0
                totalWeightSub = calcTotalWeight(subDataSet, labelIndex, True)
                prob = totalWeightSub / totalWeightV
                newEntropy += prob * calcShannonEnt(subDataSet, labelIndex)
    else:  # ������������
        uniqueValsList = list(uniqueVals)
        if 'N' in uniqueValsList:
            uniqueValsList.remove('N')
        sortedUniqueVals = sorted(uniqueValsList)  # ������ֵ����
        listPartition = []
        minEntropy = inf
        if len(sortedUniqueVals) == 1:  # ���ֻ��һ��ֵ�����Կ���ֻ�����Ӽ���û�����Ӽ�
            totalWeightLeft = calcTotalWeight(dataSet, labelIndex, True)
            probLeft = totalWeightLeft / totalWeightV
            minEntropy = probLeft * calcShannonEnt(dataSet, labelIndex)
        else:
            for j in range(len(sortedUniqueVals) - 1):  # ���㻮�ֵ�
                partValue = (float(sortedUniqueVals[j]) + float(
                    sortedUniqueVals[j + 1])) / 2
                # ��ÿ�����ֵ㣬������Ϣ��
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
                if Entropy < minEntropy:  # ȡ��С����Ϣ��
                    minEntropy = Entropy
        newEntropy = minEntropy
    gain = totalWeightV / totalWeight * (baseEntropy - newEntropy)
    return gain


def calcGainRatio(dataSet, labelIndex, labelPropertyi):
    """
    type: (list, int, int) -> float, int
    ������Ϣ������,������Ϣ�����ʺ��������ԵĻ��ֵ�
    dataSet: ���ݼ�
    labelIndex: ����ֵ����
    labelPropertyi: ����ֵ���ͣ�0Ϊ��ɢ��1Ϊ����
    """
    baseEntropy = calcShannonEnt(dataSet, labelIndex)  # ������ڵ����Ϣ��
    featList = [example[labelIndex] for example in dataSet]  # ����ֵ�б�
    uniqueVals = set(featList)  # ����������������ֵ
    newEntropy = 0.0
    bestPartValuei = None
    IV = 0.0
    totalWeight = 0.0
    totalWeightV = 0.0
    totalWeight = calcTotalWeight(dataSet, labelIndex, True)  # ������Ȩ��
    totalWeightV = calcTotalWeight(dataSet, labelIndex, False)  # �ǿ�����Ȩ��
    if labelPropertyi == 0:  # ����ɢ������
        for value in uniqueVals:  # ��ÿ������ֵ���������ݼ�, ������Ӽ�����Ϣ��
            subDataSet = splitDataSet(dataSet, labelIndex, value)
            totalWeightSub = 0.0
            totalWeightSub = calcTotalWeight(subDataSet, labelIndex, True)
            if value != 'N':
                prob = totalWeightSub / totalWeightV
                newEntropy += prob * calcShannonEnt(subDataSet, labelIndex)
            prob1 = totalWeightSub / totalWeight
            IV -= prob1 * log(prob1, 2)
    else:  # ������������
        uniqueValsList = list(uniqueVals)
        if 'N' in uniqueValsList:
            uniqueValsList.remove('N')
            # �����ֵ��������Ȩ�أ����ڼ���IV
            totalWeightN = 0.0
            dataSetNull = splitDataSet(dataSet, labelIndex, 'N')
            totalWeightN = calcTotalWeight(dataSetNull, labelIndex, True)
            probNull = totalWeightN / totalWeight
            if probNull > 0.0:
                IV += -1 * probNull * log(probNull, 2)

        sortedUniqueVals = sorted(uniqueValsList)  # ������ֵ����
        listPartition = []
        minEntropy = inf

        if len(sortedUniqueVals) == 1:  # ���ֻ��һ��ֵ�����Կ���ֻ�����Ӽ���û�����Ӽ�
            totalWeightLeft = calcTotalWeight(dataSet, labelIndex, True)
            probLeft = totalWeightLeft / totalWeightV
            minEntropy = probLeft * calcShannonEnt(dataSet, labelIndex)
            IV = -1 * probLeft * log(probLeft, 2)
        else:
            for j in range(len(sortedUniqueVals) - 1):  # ���㻮�ֵ�
                partValue = (float(sortedUniqueVals[j]) + float(
                    sortedUniqueVals[j + 1])) / 2
                # ��ÿ�����ֵ㣬������Ϣ��
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
                if Entropy < minEntropy:  # ȡ��С����Ϣ��
                    minEntropy = Entropy
                    bestPartValuei = partValue
                    probLeft1 = totalWeightLeft / totalWeight
                    probRight1 = totalWeightRight / totalWeight
                    IV += -1 * (probLeft1 * log(probLeft1, 2) + probRight1 * log(probRight1, 2))

        newEntropy = minEntropy
    gain = totalWeightV / totalWeight * (baseEntropy - newEntropy)
    if IV == 0.0:  # �������ֻ��һ��ֵ��IVΪ0��Ϊ�������Ϊ0��������С��ֵ
        IV = 0.0000000001
    gainRatio = gain / IV
    return gainRatio, bestPartValuei


# ѡ����õ����ݼ����ַ�ʽ
def chooseBestFeatureToSplit(dataSet, labelProperty):
    """
    type: (list, int) -> int, float
    :param dataSet: ������
    :param labelProperty: ����ֵ���ͣ�1 ������ 0 ��ɢ
    :return: ��ѻ������Ե��������������ԵĻ���ֵ
    """
    numFeatures = len(labelProperty)  # ������
    bestInfoGainRatio = 0.0
    bestFeature = -1
    bestPartValue = None  # ����������ֵ����ѻ���ֵ
    gainSum = 0.0
    gainAvg = 0.0
    for i in range(numFeatures):  # ��ÿ������ѭ��
        infoGain = calcGain(dataSet, i, labelProperty[i])
        gainSum += infoGain
    gainAvg = gainSum / numFeatures
    for i in range(numFeatures):  # ��ÿ������ѭ��
        infoGainRatio, bestPartValuei = calcGainRatio(dataSet, i, labelProperty[i])
        infoGain = calcGain(dataSet, i, labelProperty[i])
        if infoGainRatio > bestInfoGainRatio and infoGain > gainAvg:  # ȡ��Ϣ�������ƽ����������Ϣ��������������
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
            bestPartValue = bestPartValuei
    return bestFeature, bestPartValue


# ͨ�����򷵻س��ִ����������
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


# ������, ������ ���� �������ԣ�0 ��ɢ�� 1 ������
def createTree(dataSet, labels, labelProperty):
    classList = [example[-1] for example in dataSet]  # �������
    weightList = [example[-2] for example in dataSet]  # Ȩ������
    if classList.count(classList[0]) == len(classList):  # ���ֻ��һ����𣬷���
        totalWeiht = calcTotalWeight(dataSet,0,True)
        return (classList[0], round(totalWeiht,1),0.0)
    #totalWeight = calcTotalWeight(dataSet, 0, True)
    if len(dataSet[0]) == 1:  # ����������������������ˣ����س��ִ����������
        return majorityCnt(classList)
    bestFeat, bestPartValue = chooseBestFeatureToSplit(dataSet,
                                                       labelProperty)  # ���ŷ�������������
    if bestFeat == -1:  # ����޷�ѡ�����ŷ������������س��ִ����������
        return majorityCnt(classList, weightList)
    if labelProperty[bestFeat] == 0:  # ����ɢ������
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        labelsNew = copy.copy(labels)
        labelPropertyNew = copy.copy(labelProperty)
        del (labelsNew[bestFeat])  # �Ѿ�ѡ����������ٲ������
        del (labelPropertyNew[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueValue = set(featValues)  # ����������������ֵ
        uniqueValue.discard('N')
        for value in uniqueValue:  # ��ÿ������ֵ���ݹ鹹����
            subLabels = labelsNew[:]
            subLabelProperty = labelPropertyNew[:]
            myTree[bestFeatLabel][value] = createTree(
                splitDataSetWithNull(dataSet, bestFeat, value), subLabels,
                subLabelProperty)
    else:  # ����������������ɾ�����������ֱ𹹽���������������
        bestFeatLabel = labels[bestFeat] + '<' + str(bestPartValue)
        myTree = {bestFeatLabel: {}}
        subLabels = labels[:]
        subLabelProperty = labelProperty[:]
        # ����������
        valueLeft = 'Y'
        myTree[bestFeatLabel][valueLeft] = createTree(
            splitDataSetWithNull(dataSet, bestFeat, bestPartValue, 'L'), subLabels,
            subLabelProperty)
        # ����������
        valueRight = 'N'
        myTree[bestFeatLabel][valueRight] = createTree(
            splitDataSetWithNull(dataSet, bestFeat, bestPartValue, 'R'), subLabels,
            subLabelProperty)
    return myTree


# �����㷨
def classify(inputTree, classList, featLabels, featLabelProperties, testVec):
    firstStr = list(inputTree.keys())[0]  # ���ڵ�
    firstLabel = firstStr
    lessIndex = str(firstStr).find('<')
    if lessIndex > -1:  # ����������͵�����
        firstLabel = str(firstStr)[:lessIndex]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstLabel)  # ���ڵ��Ӧ������
    classLabel = {}
    for classI in classList:
        classLabel[classI] = 0.0
    for key in secondDict.keys():  # ��ÿ����֧ѭ��
        if featLabelProperties[featIndex] == 0:  # ��ɢ������
            if testVec[featIndex] == key:  # ������������ĳ����֧
                if type(secondDict[key]).__name__ == 'dict':  # �÷�֧����Ҷ�ӽڵ㣬�ݹ�
                    classLabelSub = classify(secondDict[key], classList, featLabels,
                                          featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[classKey]
                else:  # �����Ҷ�ӣ� ���ؽ��
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict[key][1]
                        else:
                            classLabel[classKey] += secondDict[key][2]
            elif testVec[featIndex] == 'N':  # �����������������ֵȱʧ�������ÿ����֧
                if type(secondDict[key]).__name__ == 'dict':  # �÷�֧����Ҷ�ӽڵ㣬�ݹ�
                    classLabelSub = classify(secondDict[key], classList, featLabels,
                                          featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[key]
                else:  # �����Ҷ�ӣ� ���ؽ��
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict[key][1]
                        else:
                            classLabel[classKey] += secondDict[key][2]
        else:
            partValue = float(str(firstStr)[lessIndex + 1:])
            if testVec[featIndex] == 'N':  # �����������������ֵȱʧ�����ÿ����֧�Ľ���Ӻ�
                # ����������
                if type(secondDict[key]).__name__ == 'dict':  # �÷�֧����Ҷ�ӽڵ㣬�ݹ�
                    classLabelSub = classify(secondDict[key], classList, featLabels,
                                          featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[classKey]
                else:  # �����Ҷ�ӣ� ���ؽ��
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict[key][1]
                        else:
                            classLabel[classKey] += secondDict[key][2]
            elif float(testVec[featIndex]) <= partValue and key == 'Y':  # ����������
                if type(secondDict['Y']).__name__ == 'dict':  # �÷�֧����Ҷ�ӽڵ㣬�ݹ�
                    classLabelSub = classify(secondDict['Y'], classList, featLabels,
                                             featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[classKey]
                else:  # �����Ҷ�ӣ� ���ؽ��
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict['Y'][1]
                        else:
                            classLabel[classKey] += secondDict['Y'][2]
            elif float(testVec[featIndex]) > partValue and key == 'N':
                if type(secondDict['N']).__name__ == 'dict':  # �÷�֧����Ҷ�ӽڵ㣬�ݹ�
                    classLabelSub = classify(secondDict['N'], classList, featLabels,
                                             featLabelProperties, testVec)
                    for classKey in classLabel.keys():
                        classLabel[classKey] += classLabelSub[classKey]
                else:  # �����Ҷ�ӣ� ���ؽ��
                    for classKey in classLabel.keys():
                        if classKey == secondDict[key][0]:
                            classLabel[classKey] += secondDict['N'][1]
                        else:
                            classLabel[classKey] += secondDict['N'][2]

    return classLabel


# �洢������
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


# ��ȡ������, �ļ������ڷ���None
def grabTree(filename):
    import pickle
    if os.path.isfile(filename):
        fr = open(filename)
        return pickle.load(fr)
    else:
        return None


# ���Ծ�������ȷ��
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


# ����ͶƱ�ڵ���ȷ��
def testingMajor(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major[0] != data_test[i][-1]:
            error += 1
    # print 'major %d' %error
    return float(error)


# ���֦
def postPruningTree(inputTree, classSet, dataSet, data_test, labels, labelProperties):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    classList = [example[-1] for example in dataSet]
    weightList = [example[-2] for example in dataSet]
    featkey = copy.deepcopy(firstStr)
    if '<' in firstStr:  # ������������ֵ��ʹ��������ʽ���������ǩ��value
        featkey = re.compile("(.+<)").search(firstStr).group()[:-1]
        featvalue = float(re.compile("(<.+)").search(firstStr).group()[1:])
    labelIndex = labels.index(featkey)
    temp_labels = copy.deepcopy(labels)
    temp_labelProperties = copy.deepcopy(labelProperties)
    if labelProperties[labelIndex] == 0:  # ��ɢ����
        del (labels[labelIndex])
        del (labelProperties[labelIndex])
    for key in secondDict.keys():  # ��ÿ����֧
        if type(secondDict[key]).__name__ == 'dict':  # �������Ҷ�ӽڵ�
            if temp_labelProperties[labelIndex] == 0:  # ��ɢ��
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
