import csv
import os
import torch as t
import numpy as np
from math import e
import pandas as pd
from scipy import io

# 读取CSV文件并将其转换为浮点数列表，然后返回一个PyTorch张量
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:# 打开CSV文件
        reader = csv.reader(csv_file)# 创建一个reader对象
        md_data = []# 初始化数据列表
        # 遍历CSV文件的每一行，将字符串转换为浮点数并添加到列表中
        md_data += [[float(i) for i in row] for row in reader]
        # 将列表转换为PyTorch张量
        return t.FloatTensor(md_data)

# 类似于read_csv，但是用于读取文本文件
def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)

# 读取.mat文件中的特定变量，并将其转换为PyTorch张量
def read_mat(path, name):
    matrix = io.loadmat(path)# 使用scipy.io加载.mat文件
    matrix = t.FloatTensor(matrix[name])# 获取文件中的特定变量并转换为张量
    return matrix

# 读取特定路径下的数据，并将它们组织成一个列表，每个元素是一个字典
def read_md_data(path, validation):
    result = [{} for _ in range(validation)]# 初始化结果列表
    for filename in os.listdir(path):# 遍历目录下的所有文件
        data_type = filename[filename.index('_')+1:filename.index('.')-1]# 提取数据类型
        num = int(filename[filename.index('.')-1]) # 提取编号
        # 读取CSV文件并存储
        result[num-1][data_type] = read_csv(os.path.join(path, filename))
    return result

# 从邻接矩阵生成边索引
def get_edge_index(matrix):
    edge_index = [[], []]# 初始化边索引列表
    for i in range(matrix.size(0)): # 遍历矩阵的每一行
        for j in range(matrix.size(1)):# 遍历矩阵的每一列
            if matrix[i][j] != 0:# 如果矩阵元素非零
                edge_index[0].append(i)# 添加行索引到第一个列表
                edge_index[1].append(j)# 添加列索引到第二个列表
    return t.LongTensor(edge_index) # 返回PyTorch长整型张量

# 根据邻接矩阵计算高斯矩阵Gauss_M
def Gauss_M(adj_matrix, N):
    GM = np.zeros((N, N))# 初始化高斯矩阵
    rm = N * 1. / sum(sum(adj_matrix * adj_matrix))# 计算参数rm
    for i in range(N):# 遍历矩阵的每一行
        for j in range(N):# 遍历矩阵的每一列
            # 计算高斯矩阵元素
            GM[i][j] = e ** (-rm * (np.dot(adj_matrix[i, :] - adj_matrix[j, :], adj_matrix[i, :] - adj_matrix[j, :])))
    return GM

# 根据邻接矩阵计算高斯矩阵Gauss_D，与Gauss_M类似，但是用于转置矩阵
def Gauss_D(adj_matrix, M):
    GD = np.zeros((M, M))# 初始化高斯矩阵
    T = adj_matrix.transpose()# 转置邻接矩阵
    rd = M * 1. / sum(sum(T * T))# 计算参数rd
    for i in range(M):# 遍历矩阵的每一行
        for j in range(M): # 遍历矩阵的每一列
            # 计算高斯矩阵元素
            GD[i][j] = e ** (-rd * (np.dot(T[i] - T[j], T[i] - T[j])))
    return GD

# 准备数据集的函数
def prepare_data(opt):
    dataset = {}
    # 读取CSV文件并转换为numpy数组
    dd_data = pd.read_csv(opt.data_path + '/d2d_do_file.csv', index_col=0)
    dd_mat = np.array(dd_data)

    mm_data = pd.read_csv(opt.data_path + '/p2p_smith_file.csv', index_col=0)
    mm_mat = np.array(mm_data)

    mi_dis_data = pd.read_csv(opt.data_path + '/adj_file.csv', index_col=0)

    # 初始化数据集字典，包括正样本和负样本的索引
    dataset['md_p'] = t.FloatTensor(np.array(mi_dis_data))
    dataset['md_true'] = dataset['md_p']

    # 初始化存储0值和1值索引的列表
    all_zero_index = []
    all_one_index = []
    # 遍历邻接矩阵dataset['md_p']，收集0值和1值的索引
    for i in range(dataset['md_p'].size(0)): # 行数
        for j in range(dataset['md_p'].size(1)): # 列数
            if dataset['md_p'][i][j] < 1:
                all_zero_index.append([i, j])# 添加0值索引
            if dataset['md_p'][i][j] >= 1:
                all_one_index.append([i, j]) # 添加1值索引

    # 设置随机种子以确保结果可复现
    np.random.seed(0)
    # 打乱0值和1值索引的顺序
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)

    # 截断0值索引列表，使其长度与1值索引列表相同，实现1:1正负样本划分
    all_zero_index = all_zero_index[: len(all_one_index)]

    # 将索引列表转换为PyTorch张量
    # 将索引张量分割为10份，用于10折交叉验证# 将索引张量分割为10份，用于10折交叉验证
    zero_tensor = t.LongTensor(all_zero_index)
    zero_index = zero_tensor.split(int(zero_tensor.size(0) / 10), dim=0)
    one_tensor = t.LongTensor(all_one_index)
    one_index = one_tensor.split(int(one_tensor.size(0) / 10), dim=0)

    # 将9份数据合并用于训练，剩余1份用于测试
    cross_zero_index = t.cat([zero_index[i] for i in range(9)])
    cross_one_index = t.cat([one_index[j] for j in range(9)])
    # 进一步分割测试数据，用于validation次验证
    new_zero_index = cross_zero_index.split(int(cross_zero_index.size(0) / opt.validation), dim=0)
    new_one_index = cross_one_index.split(int(cross_one_index.size(0) / opt.validation), dim=0)
    # 准备数据集字典，包含训练和测试数据的索引
    dataset['md'] = []
    for i in range(opt.validation):# 对于每一次验证
        a = [i for i in range(opt.validation)]# 初始化索引列表
        if opt.validation != 1: # 如果不是单次验证
            del a[i]# 移除当前验证的索引
        # 添加训练和测试索引到数据集字典
        dataset['md'].append({'test': [new_one_index[i], new_zero_index[i]],
                              'train': [t.cat([new_one_index[j] for j in a]), t.cat([new_zero_index[j] for j in a])]})

    # 准备独立的测试集，使用最后两份数据
    dataset['independent'] = []
    in_zero_index_test = zero_index[-2]
    in_one_index_test = one_index[-2]
    dataset['independent'].append({'test': [in_one_index_test, in_zero_index_test],
                                   'train': [cross_one_index,cross_zero_index]})

    # 计算邻接矩阵的高斯相似性矩阵DGSM和MGSM
    DGSM = Gauss_D(dataset['md_p'].numpy(), dataset['md_p'].size(1))
    MGSM = Gauss_M(dataset['md_p'].numpy(), dataset['md_p'].size(0))
    nd = mi_dis_data.shape[1]
    nm = mi_dis_data.shape[0]

    # 根据高斯相似性矩阵和原始相似性矩阵，计算最终的ID和IM矩阵
    ID = np.zeros([nd, nd])# 初始化ID矩阵

    for h1 in range(nd): # 遍历微米数据的列
        for h2 in range(nd):# 遍历行
            if dd_mat[h1, h2] == 0: # 如果原始相似性为0
                ID[h1, h2] = DGSM[h1, h2]# 使用高斯相似性
            else:
                ID[h1, h2] = (dd_mat[h1, h2] + DGSM[h1, h2]) / 2# 否则取平均

    IM = np.zeros([nm, nm])# 初始化IM矩阵

    for q1 in range(nm):# 遍历药物数据的行
        for q2 in range(nm): # 遍历列
            if mm_mat[q1, q2] == 0:# 如果原始相似性为0
                IM[q1, q2] = MGSM[q1, q2]# 使用高斯相似性
            else:
                IM[q1, q2] = (mm_mat[q1, q2] + MGSM[q1, q2]) / 2# 否则取平均
    # 将ID和IM矩阵添加到数据集字典，并转换为PyTorch张量
    dataset['ID'] = t.from_numpy(ID)
    dataset['IM'] = t.from_numpy(IM)

    return dataset