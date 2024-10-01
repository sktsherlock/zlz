import numpy as np
import torch
from sklearn.cluster import KMeans


# 超边矩阵拼接函数
def hyperedge_concat(*H_list):
    H = None  # 初始化超边矩阵
    for h in H_list:
        # 检查h是否非空
        if h is not None and len(h) != 0:
            # 如果是第一个超边矩阵，则直接赋值
            if H is None:
                H = h
            else:
                # 如果h不是列表，则水平堆叠
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    # 如果h是列表，则逐元素水平堆叠
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


# 从聚类结果构建边列表的辅助函数
def _construct_edge_list_from_cluster(X, clusters):
    """
    从单一模态数据的聚类结果构建边列表（numpy数组）。

    :param X: 特征矩阵
    :param clusters: 聚类的数量
    :return: 超边矩阵H
    """
    N = X.shape[0]  # 样本数量
    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=0).fit(X)
    assignment = kmeans.labels_  # 聚类标签

    H = np.zeros([N, clusters])  # 初始化超边矩阵
    for i in range(N):
        # 将对应聚类的行置为1
        H[i, assignment[i]] = 1

    return H
# 使用KMeans构建超边矩阵的函数
def construct_H_with_Kmeans(X, clusters, split_diff_scale=False):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])  # 确保X是二维数组

    if type(clusters) == int:
        clusters = [clusters]  # 如果clusters是整数，则将其转换为单元素列表

    H = []
    for c in clusters:
        H_tmp = _construct_edge_list_from_cluster(X, c)  # 为每个clusters值构建H
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)  # 拼接超边矩阵
        else:
            H.append(H_tmp)  # 将每个H_tmp添加到列表中
    return H


# 从超边矩阵生成图的邻接矩阵的辅助函数
def _generate_G_from_H(H, variable_weight=False):
    # 将输入的H转换为NumPy数组
    H = np.array(H)
    # 获取超边的数量
    n_edge = H.shape[1]
    # the weight of the hyperedge
    # 超边的权重初始化为1
    W = np.ones(n_edge)
    # 计算度矩阵DV，即每一行的和
    DV = np.sum(H * W, axis=1)
    # 计算度矩阵DE，即每一列的和
    DE = np.sum(H, axis=0)
    # 计算DE的逆矩阵，用于后续的规范化
    invDE = np.mat(np.diag(np.power(DE, -1)))
    # 计算DV的0.5次幂的对角矩阵
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))

    # 将W和H转换为矩阵
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    # 如果需要变量权重，则返回DV的0.5次幂与H的乘积，以及DE的逆矩阵与H.T的乘积
    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        # 计算规范化的邻接矩阵G
        G = DV2 * H * W * invDE * HT * DV2
        G = torch.Tensor(G)
        return G


# 根据超边矩阵生成图的邻接矩阵的函数
def generate_G_from_H(H, variable_weight=False):
    # 如果输入H不是列表，则直接调用_generate_G_from_H函数
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        # 如果输入H是列表，则遍历每个子矩阵，并递归调用generate_G_from_H函数
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G
