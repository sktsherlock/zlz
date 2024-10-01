from torch import nn
from param import parameter_parser
# 解析命令行参数或配置文件中的参数
args = parameter_parser()


class Myloss(nn.Module):
    """
     自定义损失函数类，继承自PyTorch的nn.Module。

     这个类定义了一个用于模型训练的自定义损失函数，
     它可能对正样本和负样本赋予不同的权重。
     """
    def __init__(self):
        """
            初始化Myloss类的实例。
        """
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, input, target):
        """
        计算损失函数的前向传播。

        参数:
        one_index : 正样本的索引。
        zero_index : 负样本的索引。
        input : 模型的预测输出。
        target : 目标值，通常是真实标签。

        返回:
        加权后的损失函数的总和。
        """
        # 初始化均方误差损失函数，设置reduction='none'以保留每个样本的损失值
        loss = nn.MSELoss(reduction='none')
        # 计算输入和目标之间的损失
        loss_sum = loss(input, target)
        # 根据正负样本的索引以及alpha参数计算加权损失
        return (1-args.alpha)*loss_sum[one_index].sum() + args.alpha*loss_sum[zero_index].sum()


def get_L2reg(parameters):
    """
    计算L2正则化项。

    参数:
    parameters : 模型的参数列表。

    返回:
    L2正则化项的和。
    """
    reg = 0
    # 遍历模型的所有参数
    for param in parameters:
        # 计算每个参数的L2范数并累加到reg变量中
        reg += 0.5 * (param ** 2).sum()
    return reg