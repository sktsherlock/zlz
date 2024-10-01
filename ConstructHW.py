
import math
import numpy as np
import hypergraph_construct_KNN
import hypergraph_construct_kmeans

def constructHW_knn(X,K_neigs,is_probH):

    """incidence matrix"""
    # 使用K近邻算法构建超图的关联矩阵H
    H = hypergraph_construct_KNN.construct_H_with_KNN(X,K_neigs,is_probH)
    # 根据关联矩阵H生成超图的邻接矩阵G
    G = hypergraph_construct_KNN._generate_G_from_H(H)

    return G

def constructHW_kmean(X,clusters):

    """incidence matrix"""
    # 使用K均值聚类算法构建超图的关联矩阵H
    H = hypergraph_construct_kmeans.construct_H_with_Kmeans(X,clusters)
    # 根据关联矩阵H生成超图的邻接矩阵G
    G = hypergraph_construct_kmeans._generate_G_from_H(H)

    return G
