
import numpy as np
import torch
import math

# 计算欧式距离矩阵
def Eu_dis(x):
    x = np.mat(x)  # 将输入转换为矩阵
    aa = np.sum(np.multiply(x, x), 1)  # 计算每行的平方和
    ab = x * x.T  # 计算矩阵与其转置的乘积
    dist_mat = aa + aa.T - 2 * ab  # 根据欧式距离公式计算距离矩阵
    dist_mat[dist_mat < 0] = 0  # 将负数设置为0（可选）
    dist_mat = np.sqrt(dist_mat)  # 开平方根得到实际距离
    dist_mat = np.maximum(dist_mat, dist_mat.T)  # 取行与列的最大值作为距离（可选）
    return dist_mat

# 特征矩阵拼接函数
def feature_concat(*F_list, normal_col=False):
    features = None  # 初始化特征矩阵
    for f in F_list:  # 遍历输入的特征列表
        if f is not None and f != []:  # 确保特征矩阵不为空
            if len(f.shape) > 2:  # 如果特征矩阵维度大于2，则将其重塑为二维
                f = f.reshape(-1, f.shape[-1])
            if normal_col:  # 如果需要列归一化
                f_max = np.max(np.abs(f), axis=0)  # 计算每列的最大值
                f = f / f_max  # 归一化
            if features is None:  # 如果是第一个特征矩阵
                features = f
            else:  # 否则，水平拼接特征矩阵
                features = np.hstack((features, f))
    if normal_col:  # 如果需要整体归一化
        features_max = np.max(np.abs(features), axis=0)  # 计算整个特征矩阵的最大值
        features = features / features_max  # 归一化
    return features

# 超边矩阵拼接函数
def hyperedge_concat(*H_list):
    H = None  # 初始化超边矩阵
    for h in H_list:  # 遍历输入的超边列表
        if h is not None and len(h) != 0:  # 确保超边矩阵不为空
            if H is None:  # 如果是第一个超边矩阵
                H = h
            else:  # 否则，根据h的类型进行拼接
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):  # 如果h是列表，则逐元素拼接
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H

# 根据超边矩阵生成图的邻接矩阵
def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:  # 如果H不是列表，则直接生成邻接矩阵
        return _generate_G_from_H(H, variable_weight)
    else:  # 如果H是列表，则对每个子H生成邻接矩阵
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

# 生成图的邻接矩阵的内部函数
def _generate_G_from_H(H, variable_weight=False):
    # 将输入的H转换为NumPy数组
    H = np.array(H)
    # 获取超边的数量
    n_edge = H.shape[1]
    # the weight of the hyperedge
    # 初始化超边权重为1
    W = np.ones(n_edge)
    # 计算度矩阵DV（每一行的和）
    DV = np.sum(H * W, axis=1)
    # 计算度矩阵DE（每一列的和）
    DE = np.sum(H, axis=0)
    # 计算DE的逆矩阵，用于后续的规范化
    invDE = np.mat(np.diag(np.power(DE, -1)))
    # 计算DV的0.5次幂的对角矩阵
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    # 将W转换为对角矩阵
    W = np.mat(np.diag(W))
    # 将H转换为矩阵
    H = np.mat(H)
    # 获取H的转置
    HT = H.T
    # 如果需要变量权重，则返回DV的0.5次幂与H的乘积，以及DE的逆矩阵与H.T的乘积
    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        # 计算规范化的邻接矩阵G
        G = DV2 * H * W * invDE * HT * DV2
        # 将结果转换为PyTorch张量
        G = torch.Tensor(G)
        return G

# 根据K近邻和距离矩阵构建超边矩阵
def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    # 获取对象的数量
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    # 初始化超边矩阵H为零矩阵
    H = np.zeros((n_obj, n_edge))
    # 遍历每个对象，构建与其相关的超边
    for center_idx in range(n_obj):
        # 将自身与自身的距离设置为0
        dis_mat[center_idx, center_idx] = 0
        # 获取距离向量
        dis_vec = dis_mat[center_idx]
        # 获取最近的k_neig个邻居的索引
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        # 计算距离向量的平均值
        avg_dis = np.average(dis_vec)
        # 确保中心索引在最近邻居中
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx
        # 为每个最近邻居构建超边
        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                # 如果是概率型超图，则使用高斯核函数计算超边权重
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2) ## affinity matrix的计算公式
            else:
                # 如果不是概率型超图，则超边权重为1
                H[node_idx, center_idx] = 1.0
    return H

# 根据K近邻和距离矩阵构建超边矩阵
def construct_H_with_KNN(X, K_neigs, split_diff_scale=False, is_probH=False, m_prob=1):
    # 确保X是二维数组
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    # 确保K_neigs是列表
    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    # 计算欧氏距离矩阵
    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        # 为每个k_neig构建超边矩阵
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        # 如果不需要区分不同的尺度，则将不同k_neig的超边矩阵连接起来
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            # 如果需要区分不同的尺度，则将不同k_neig的超边矩阵存储在列表中
            H.append(H_tmp)
    return H