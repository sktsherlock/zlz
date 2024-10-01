import torch
from prepareData import prepare_data
import numpy as np
from torch import optim
from param import parameter_parser
from Module import HGCLAMIR
from utils import get_L2reg, Myloss
from Calculate_Metrics import Metric_fun
from trainData import Dataset
import ConstructHW
import wandb
from matplotlib import pyplot

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义训练一个epoch的函数
def train_epoch(model, train_data, optim, opt):
    # 设置模型为训练模式
    model.train()
    # 初始化回归损失函数
    regression_crit = Myloss()
    # 从训练数据中获取正样本和负样本的索引
    one_index = train_data[2][0].to(device).t().tolist()
    zero_index = train_data[2][1].to(device).t().tolist()
    # 将训练数据中的相似性数据转换为张量并移动到设备（GPU或CPU）
    dis_sim_integrate_tensor = train_data[0].to(device)
    mi_sim_integrate_tensor = train_data[1].to(device)

    # 将miRNA数据和相似性数据合并，然后转换为张量并移动到设备
    concat_miRNA = np.hstack(
        [train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
    concat_mi_tensor = torch.FloatTensor(concat_miRNA)
    concat_mi_tensor = concat_mi_tensor.to(device)

    # 使用自定义的函数构建knn和kmean图，并将结果转换为张量并移动到设备
    G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
    G_mi_Km = ConstructHW.constructHW_kmean(concat_mi_tensor.detach().cpu().numpy(), clusters=[9])
    G_mi_Kn = G_mi_Kn.to(device)
    G_mi_Km = G_mi_Km.to(device)

    # 将疾病数据和相似性数据合并，然后转换为张量并移动到设备
    concat_dis = np.hstack(
        [train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
    concat_dis_tensor = torch.FloatTensor(concat_dis)
    concat_dis_tensor = concat_dis_tensor.to(device)

    # 使用自定义的函数构建knn和kmean图，并将结果转换为张量并移动到设备
    G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
    G_dis_Km = ConstructHW.constructHW_kmean(concat_dis_tensor.detach().cpu().numpy(), clusters=[9])
    G_dis_Kn = G_dis_Kn.to(device)
    G_dis_Km = G_dis_Km.to(device)

    # 循环进行训练，直到达到指定的周期数
    for epoch in range(1, opt.epoch+1):
        # 调用模型进行前向传播，获取分数以及分类损失
        score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
                                               G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
        # 计算回归损失
        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
        # 计算L2正则化损失
        reg_loss = get_L2reg(model.parameters())
        # 总损失是回归损失、分类损失和正则化损失的和
        tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss
        # 清空之前的梯度
        optim.zero_grad()
        # 反向传播，计算当前损失相对于模型参数的梯度
        tol_loss.backward()
        # 更新模型参数
        optim.step()
    # 在训练结束后，调用测试函数评估模型性能
    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(model, train_data, concat_mi_tensor, concat_dis_tensor,
                                 G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
    # 返回测试结果
    return true_value_one, true_value_zero, pre_value_one, pre_value_zero


# 定义测试函数，评估模型性能
def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
    # 设置模型为评估模式，这样在前向传播时不会计算梯度
    model.eval()
    # 调用模型进行前向传播，获取分数，忽略分类损失
    score,_,_ = model(concat_mi_tensor, concat_dis_tensor,
                      G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
    # 获取测试数据中正样本和负样本的索引
    test_one_index = data[3][0].t().tolist()
    test_zero_index = data[3][1].t().tolist()
    # 获取测试数据中正样本和负样本的真实值
    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]
    # 根据索引获取模型预测的正样本和负样本的分数
    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]
    # 返回测试集上的真值和预测值
    return true_one, true_zero, pre_one, pre_zero


# 定义评估函数evaluate，计算性能指标
def evaluate(true_one, true_zero, pre_one, pre_zero, i):
    # 初始化度量工具，Metric_fun是一个已经定义好的类，用于计算性能指标
    Metric = Metric_fun()
    # 初始化一个张量（tensor）来存储性能指标，这里存储7个不同的性能指标
    metrics_tensor = np.zeros((1, 7))

    # test_po_num是true_one数组的长度，即正样本的数量
    test_po_num = true_one.shape[0]
    # 创建一个布尔索引数组，用于找到true_zero中值为0的位置，即负样本
    test_index = np.array(np.where(true_zero == 0))
    # 设置随机种子以确保结果的可重复性
    # np.random.seed(seed)
    # 打乱索引数组，以实现随机抽样
    np.random.shuffle(test_index.T)
    # 选择前test_po_num个索引作为测试集的负样本索引
    test_ne_index = tuple(test_index[:, :test_po_num])

    # 根据测试集的负样本索引，从true_zero中选择对应的负样本
    eval_true_zero = true_zero[test_ne_index]
    # 将正样本和负样本合并为一个数组
    eval_true_data = torch.cat([true_one, eval_true_zero])

    # 同样地，根据测试集的负样本索引，从pre_zero中选择对应的预测负样本
    eval_pre_zero = pre_zero[test_ne_index]
    # 将预测的正样本和负样本合并为一个数组
    eval_pre_data = torch.cat([pre_one, eval_pre_zero])

    # 使用Metric对象的cv_mat_model_evaluate方法计算性能指标
    # print('  epcho '+str(seed)+' auc, aupr, f1_score, accuracy, recall, specificity, precision ')
    # print(Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data))
    # 并将结果累加到metrics_tensor中
    metrics_tensor = metrics_tensor + Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data, i)
    print(Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data, i))

    # 返回平均性能指标
    return metrics_tensor


# 主函数，控制整个训练和评估流程
def main(opt):
    # 准备数据集
    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)

    # 初始化交叉验证性能指标数组
    metrics_cross = np.zeros((1, 7))
    # 遍历验证集的每一个样本
    for i in range(opt.validation):
        # 初始化模型和优化器
        # hidden_list: 定义模型隐藏层的大小列表
        hidden_list = [256, 256]  # 这里定义了两层隐藏层，每层大小为256
        # num_proj_hidden: 定义投影层隐藏层的大小
        num_proj_hidden = 64

        # 创建模型实例，这里假设HGCLAMIR是一个已经定义好的模型类
        # args.mi_num, args.dis_num, args是模型需要的一些参数
        model = HGCLAMIR(args.pi_num, args.dis_num, hidden_list, num_proj_hidden, args)
        # 将模型移动到指定的设备（如GPU）上
        model.to(device)
        # 训练模型并获取测试结果
        # optimizer: 定义优化器，这里使用的是Adam优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率设置为0.0001
        # train_epoch是一个函数，用于训练模型一个epoch并返回一些指标
        # train_data[i]是当前样本的数据
        # opt是包含其他配置选项的对象
        true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(
            model, train_data[i], optimizer, opt
        )
        # 计算性能指标并累加
        # evaluate是一个函数，用于根据模型输出和真实标签计算性能指标
        metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero, i)
        # metrics_cross用于累积性能指标
        metrics_cross = metrics_cross + metrics_value
        print('fold '+str(i)+' auc , aupr , f1_score , accuracy , recall , specificity , precision')
        wandb.log({'metrics_cross': metrics_cross, 'metrics_value': metrics_value})
        print(metrics_value)
    # 完成所有样本的训练和评估后，计算平均性能指标
    # metrics_cross_avg是所有样本性能指标的平均值
    metrics_cross_avg = metrics_cross / 5
    # np.savetxt(f'result/avg_x_ROC.txt', metrics_cross_avg[0][7])
    # np.savetxt(f'result/avg_y_ROC.txt', metrics_cross_avg[0][8])
    # np.savetxt(f'result/avg_x_PR.txt', metrics_cross_avg[0][9])
    # np.savetxt(f'result/avg_y_PR.txt', metrics_cross_avg[0][10])
    # 打印平均性能指标的结果
    print('All_metrics_avg:auc , aupr , f1_score , accuracy , recall , specificity , precision ')
    print(metrics_cross_avg)


if __name__ == '__main__':
    # 解析命令行参数
    args = parameter_parser()
    wandb.init(config=args, reinit=True)
    # 调用主函数
    main(args)

