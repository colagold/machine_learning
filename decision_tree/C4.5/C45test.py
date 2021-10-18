import C45
import treePlotter

# 读取数据文件
fr = open(r'PlayData.txt')
# 生成数据集
lDataSet = [inst.strip().split('\t') for inst in fr.readlines()]
print(lDataSet)
# 样本特征标签
labels = ['天气', '周末', '促销']
# 样本特征类型，0为离散，1为连续
labelProperties = [0, 0, 0]
# 类别向量
classList = ['是', '否']
# 验证集
dataSet_test = [['rain', '72', '95', 'false', 1, 'Don’t Play'], ['rain', '72', '90', 'true', 1, 'Play']]
# 构建决策树
trees = C45.createTree(lDataSet, labels, labelProperties)
print(trees)
# 绘制决策树
treePlotter.createPlot(trees)
# 利用验证集对决策树剪枝
C45.postPruningTree(trees, classList, lDataSet, dataSet_test, labels, labelProperties)
# 绘制剪枝后的决策树
treePlotter.createPlot(trees)
# 重新赋值类别标签和类型
labels = ['天气', '周末', '促销']
labelProperties = [0, 0, 0]
# 测试样本
testVec = ['sunny', 70, 'N', 'false']
# 对测试样本分类
classLabel = C45.classify(trees, classList, labels, labelProperties, testVec)
# 打印测试样本的分类结果
print(classLabel)


