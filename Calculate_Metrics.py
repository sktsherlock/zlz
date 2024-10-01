import numpy as np
from matplotlib import pyplot

from sklearn.metrics import *
import matplotlib.pyplot as plt


class Metric_fun(object):
    def __init__(self):
        # 初始化函数，从object基类继承
        super(Metric_fun).__init__()

    def cv_mat_model_evaluate(self, association_mat, predict_mat, i):
        # 评估模型函数，接受关联矩阵和预测矩阵作为输入
        # 将输入的PyTorch张量转换成NumPy数组，并展平成一维数组
        real_score = np.mat(association_mat.detach().cpu().numpy().flatten())
        predict_score = np.mat(predict_mat.detach().cpu().numpy().flatten())
        # 调用get_metrics函数计算评估指标
        return self.get_metrics(real_score, predict_score, i)

    def get_metrics(self, real_score, predict_score, i):
        # 计算评估指标的函数，接受实际得分和预测得分作为输入

        # 对预测得分进行排序并去重
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        # 获取去重后预测得分的数量
        sorted_predict_score_num = len(sorted_predict_score)
        # 根据得分分布计算阈值
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]
        # 将预测得分复制到与阈值矩阵相同形状的矩阵中
        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1
        # 计算真正例（TP）、假正例（FP）、假负例（FN）和真负例（TN）
        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN
        # 计算假正例率（FPR）和真正例率（TPR）
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T
        # 计算ROC曲线下的面积（AUC）
        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        # plt.plot(fpr, tpr, label=auc)
        # plt.legend()
        # plt.show()

        np.savetxt(f'result0/fpr_{i}.txt', x_ROC)
        np.savetxt(f'result0/tpr_{i}.txt', y_ROC)


        # 计算召回率（recall）列表和精确率（precision）列表
        recall_list = tpr
        precision_list = TP / (TP + FP)
        # 准备PR曲线的数据点
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        # 计算PR曲线下的面积（AUPR）
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        # plt.plot(x_PR, y_PR, label=aupr)
        # plt.title('AUPR')
        # plt.ylim([0, 1])
        # plt.legend()
        # plt.show()

        np.savetxt(f'result0/recalls_{i}.txt', x_PR)
        np.savetxt(f'result0/precisions_{i}.txt', y_PR)

        # 计算F1分数列表、准确率（accuracy）列表、特异性（specificity）列表
        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)
        # 找到F1分数最大的索引，并获取最佳的F1分数、准确率、特异性、召回率和精确率
        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]

        # 返回计算得到的评估指标
        return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]
