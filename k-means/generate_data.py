import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
def get_data_set(sampleNo,mu1):
    plt.figure(0)
    #sampleNo样本个数
    #mu1均值数组
    #Sigma1方差
    data_set=[]
    for i in mu1:
        x=np.array(i)
        Sigma1 = np.array([[3, 1.5], [1.5, 3]])
        R = cholesky(Sigma1)
        s1 = np.dot(np.random.randn(sampleNo, 2), R) + x
        plt.scatter(s1[:, 0], s1[:, 1])
        data_set.extend(s1)
    plt.show()
    return get_list(data_set)

def get_list(np_list):
    return [i.tolist() for i in np_list]

