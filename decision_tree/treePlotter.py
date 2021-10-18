# -*- coding: cp936 -*-
import matplotlib.pyplot as plt

# 设置决策节点和叶节点的边框形状、边距和透明度，以及箭头的形状
decisionNode = dict(boxstyle="square,pad=0.5", fc="0.9")
leafNode = dict(boxstyle="round4, pad=0.5", fc="0.9")
arrow_args = dict(arrowstyle="<-", connectionstyle="arc3", shrinkA=0,
                  shrinkB=16)


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    newTxt = nodeTxt
    if type(nodeTxt).__name__ == 'tuple':
        newTxt = nodeTxt[0] + '\n'
        for strI in nodeTxt[1:-1]:
            newTxt += str(strI) + ','
        newTxt+= str(nodeTxt[-1])

    createPlot.ax1.annotate(newTxt, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="top", ha="center", bbox=nodeType,
                            arrowprops=arrow_args)


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {
            'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                     cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree)) + 0.5
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def createPlot0():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    # plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plotNode('叶节点', (0.8, 0.1), (0.3, 0.75), leafNode)

    '''an1 = createPlot.ax1.annotate(unicode("决策节点", 'cp936'), xy=(0.5, 0.5),
                                  xycoords="data",
                                  va="center", ha="center",
                                  bbox=dict(boxstyle="round", fc="w"))

    createPlot.ax1.annotate(unicode('叶节点', 'cp936'),
                            xytext=(0.2, 0.3), arrowprops=dict(arrowstyle="<-"),
                            xycoords=an1,
                            textcoords='axes fraction',
                            va="bottom", ha="left",
                            bbox=leafNode)'''

    an1 = createPlot.ax1.annotate("Test 1", xy=(0.5, 0.5), xycoords="data",
                                  va="center", ha="center",
                                  bbox=dict(boxstyle="round", fc="w"))
    an2 = createPlot.ax1.annotate("Test 2", xy=(0, 0.5), xycoords=an1,
                                  # (1,0.5) of the an1's bbox
                                  xytext=(-50, -50), textcoords="offset points",
                                  va="center", ha="center",
                                  bbox=dict(boxstyle="round", fc="w"),
                                  arrowprops=dict(arrowstyle="<-"))

    plt.show()


'''
    an1 = createPlot.ax1.annotate(unicode('决策节点', 'cp936'), xy=(0.5, 0.6),
                            xycoords='axes fraction',
                            textcoords='axes fraction',
                            va="bottom", ha="center", bbox=decisionNode)

    createPlot.ax1.annotate(unicode('叶节点', 'cp936'), xy=(0.8, 0.1),
                            xycoords=an1,
                            textcoords='axes fraction',
                            va="bottom", ha="center", bbox=leafNode,
                            arrowprops=arrow_args)
'''

if __name__ == '__main__':
    createPlot0()
