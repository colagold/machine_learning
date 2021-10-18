import math
def cal_res_probability(data_set):
    label=[]
    for i in data_set:
        label.append(i[-1])
    num_dicts={}
    dicts={}
    #计数
    for i in label:
        if i not in num_dicts.keys():
            num_dicts[i] = 1
        else:
            num_dicts[i] +=1

    #计算概率
    for i in num_dicts.keys():
        dicts[i]=num_dicts[i]/len(label)
    return dicts,num_dicts

def union_probability(labal_data,forecast_data,data_set,discrete_num):
    '''
    :param total_dict:
    :param forecast_data: 预测数据
    :param data_set: 数据集
    :param discrete_num: 离散数据的数量
    :return:
    '''
    probability=[]
    time=0
    for i in forecast_data:
        count=0
        if time==6:
            break
        time+=1
        for j in labal_data:#下标
            if i in data_set[j]:
                count+=1
        probability.append(float(count)/len(labal_data))
    return probability

def cal_union_probability(label_data,forecast_data,data_set,discrete_num):
    '''
    :param total_dict:
    :param forecast_data:
    :param data_set:
    :param discrete_num:
    :return:计算连续数据的联合概率
    '''
    b=[]
    index=-2
    u_list=[]
    for i in range(discrete_num):
        sum=0
        for j in label_data:
            tmp =float(data_set[j][index])
            #计算u
            sum+=tmp
        index-=1
        u_list.append(sum/len(label_data))
    index = -2
    for i in range(discrete_num):
        sum=0
        for j in label_data:
            tmp = float(data_set[j][index])
            # 计算u
            sum+=((tmp-u_list[i])*(tmp-u_list[i]))
        index -= 1
        b.append((sum/len(label_data))**0.5)
    union_probability=[]
    index=-1
    for i in range(len(u_list)):
        probability=(1/(((2*math.pi)**0.5)*b[i])*math.exp((-1)*((float(forecast_data[index])-u_list[i])**2)/(2*(b[i])**2)))
        union_probability.append(probability)
        index-=1
    return union_probability

def predict(predict_data):
    data_set=load_data()
    label_data, label_list = split_data(data_set)
    scatter_probability_list=[]
    union_probability_lists=[]
    for i in range(len(label_data)):
        scatter_probability=union_probability(label_data[i], predict_data, data_set, 6)
        # print("测试")
        # print(scatter_probability)
        union_probability_list=cal_union_probability(label_data[i], predict_data, data_set, 2)
        scatter_probability_list.append(scatter_probability)
        union_probability_lists.append(union_probability_list)
    print("离散变量概率列表")
    print(scatter_probability_list)
    print("连续变量概率列表")
    print(union_probability_lists)
    result_list=[]
    for i in range(len(label_list)):
        p=1
        for j in range(6):
            p*=scatter_probability_list[i][j]
        for j in range(2):
            p*=union_probability_lists[i][j]
        result_list.append(p)
    print("预测概率")
    print(label_list)
    print(result_list)
    print(predict_data,end='')
    print(" 此瓜是否是一个好瓜？")
    return label_list[get_max_index(max(result_list),result_list)]

def get_max_index(res,lists):
    for i in range(len(lists)):
        if res == lists[i]:
            return i
    return 0

def load_data():
    file=open('data.txt','r')
    data_set=[]
    for i in file.readlines():
        data_set.append(i.strip().split(',')[1:])
    return data_set

def split_data(data_set):
    '''
    :param data_set:
    :return:
    根据标签将数据分组
    '''
    label_list=[]
    for i in data_set:
        if i[-1] not in label_list:
            label_list.append(i[-1])
    data=[[] for i in label_list]
    k=0
    for i in data_set:
        for j in range(len(label_list)):
            if i[-1] ==label_list[j]:
                data[j].append(k)
        k+=1
    return data,label_list
if __name__=='__main__':
    predict_data=['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460]
    print(predict(predict_data))