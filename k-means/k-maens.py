import random
import numpy as np
import matplotlib.pyplot as plt
import generate_data


def distance(x, u):
    x=np.array(x)
    u=np.array(u)
    return np.linalg.norm(x - u)

def get_cluster(k, dataset):
    k_point=random_point(k,dataset) #初始化类中心，以后类中心都存储在这
    cluster=[[] for i in range(k)]
    for i in range(len(dataset)):  ##遍历所有的点
        distances=[]
        for j in range(k):
            distances.append(distance(dataset[i],k_point[j]))  #计算到中心点的距离
        index=np.argmin(distances) #划分类别
        cluster[index].append(dataset[i]) #划分到对应的类别
    return cluster,k_point

def get_new_cluster(k_point,dataset):
    k=len(k_point)
    cluster = [[] for i in range(k)]
    for i in range(len(dataset)):  ##遍历所有的点
        distances = []
        for j in range(k):
            distances.append(distance(dataset[i], k_point[j]))  # 计算到中心点的距离
        index = np.argmin(distances)  # 划分类别
        cluster[index].append(dataset[i])  # 划分到对应的类别
    return cluster


def random_point(k, dataset):
    #随机点生成
    points = []
    size = len(dataset)
    for i in range(k):
        points.append(dataset[int(size*random.random())])
    return points


def get_avg_vector(cluster_list,k_point):
    #cluster_list 每个类的列表,格式[[]]   k_point:k*element_size
    k=len(k_point)
    new_point=[[] for i in range(k)]
    element_size=len(cluster_list[0][0])
    for i in range(k): #类别
        tmp=[]
        for j in range(element_size): #维度
            value=0
            for m in range(len(cluster_list[i])): #具体值,遍历每个类别下的样本
                value+=cluster_list[i][m][j]
            value=value/len(cluster_list[i])
            tmp.append(value)
        new_point[i]=tmp
    return new_point

def is_equal(x,y):
    for i in range(len(x)):
        for j in range(len(x[0])):
            print(x[i][j])
            print(y[i][j])
            if x[i][j] != y[i][j]:
                return False
    return True

def k_means(k,dataset):
    cluster, k_point = get_cluster(k, dataset)
    new_point = get_avg_vector(cluster, k_point)
    while(k_point!=new_point):
        k_point=new_point
        new_cluster=get_new_cluster(new_point,dataset)
        new_point=get_avg_vector(new_cluster,k_point)

    print(new_point)
    print(new_cluster)
    return new_cluster,new_point

def load_data(path):
    res_list=[]
    with open(path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            line = list(map(float,line.strip("\n").split(' '))) # 去掉列表中每一个元素的换行符
            print(line)
            res_list.append(line)
    return res_list

if __name__  == '__main__':
    k=int(input("请输入k值："))
    #[[6, 6], [-6, -6], [6, -6]]
    #[[6,6],[3,3],[6,3]]
    # 读取文本数据
    dataset=load_data("data.txt")

    #自定义生成数据
    # dataset=generate_data.get_data_set(100,[[6,6],[3,3],[6,3]])
    cluster,point=k_means(k,dataset)
    print(cluster)
    x=[[] for i in range(k) ]
    y=[[] for i in range(k)]
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            x[i].append(cluster[i][j][0])
            y[i].append(cluster[i][j][1])
    fig = plt.figure()
    ax = plt.subplot()
    for i in range(k):
        ax.scatter(x[i], y[i], linewidths=[3])
        ax.scatter(point[i][0], point[i][1])
    plt.show()
