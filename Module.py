import random
import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import VariLengthInputLayer, EncodeLayer, FeedForwardLayer

# 设置设备，优先使用GPU，如果没有GPU，则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义一个函数来设置随机种子，以确保结果的可复现性
def seed_torch(seed):
    # 设置Python内置库的随机种子
    random.seed(seed)
    # 设置环境变量PYTHONHASHSEED，以确保hash随机性
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch的CPU随机种子
    torch.manual_seed(seed)
    # 设置PyTorch的GPU随机种子
    torch.cuda.manual_seed(seed)
    # 设置所有GPU的随机种子
    torch.cuda.manual_seed_all(seed)
# 设置随机种子以保证结果的可复现性
seed_torch(seed=1234)

# Transformer 编码器模块，用于处理序列数据
class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(TransformerEncoder, self).__init__()
        # 初始化参数和层
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden# 查询（Query）维度
        self.d_k = hyperpm.n_hidden# 键（Key）维度
        self.d_v = hyperpm.n_hidden# 值（Value）维度
        self.n_head = hyperpm.n_head# 注意力机制的头数
        self.dropout = hyperpm.dropout# Dropout比例
        self.n_layer = hyperpm.nlayer# 编码器层数
        self.modal_num = hyperpm.nmodal# 模态数量
        self.d_out = self.d_v * self.n_head * self.modal_num# 输出维度
        # 初始化可变长度输入层
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        # 初始化编码器层列表和前馈网络层列表
        self.Encoder = []
        self.FeedForward = []
        # 循环创建编码器层和前馈网络层
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)# 添加编码器层
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)# 添加前馈网络层

            self.FeedForward.append(feedforward)

    # 前向传播函数
    def forward(self, x):
        bs = x.size(0)# 获取批量大小
        attn_map = []# 初始化注意力图谱列表
        # 通过输入层
        x, _attn = self.InputLayer(x)

        attn = _attn.mean(dim=1)# 计算平均注意力分数
        attn_map.append(attn.detach().cpu().numpy())# 将注意力分数添加到列表中
        # 循环遍历编码器层和前馈网络层
        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)# 编码器层处理
            attn = _attn.mean(dim=1)# 计算平均注意力分数
            x = self.FeedForward[i](x)# 前馈网络层处理
            attn_map.append(attn.detach().cpu().numpy())# 将注意力分数添加到列表中
        # 调整输出形状
        x = x.view(bs, -1)

        # output = self.Outputlayer(x)
        # 返回编码器的输出
        return x

# 图卷积网络模块，用于处理图数据
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()# 调用基类的构造函数
        # 初始化图卷积层的输入和输出特征数量
        self.in_features = in_ft
        self.out_features = out_ft
        # 定义权重参数，其维度为 [输入特征数, 输出特征数]
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        # 如果bias为False，则注册一个不存在的偏置参数
        if bias:

            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            # 如果bias为False，则注册一个不存在的偏置参数
            self.register_parameter('bias', None)
        # 调用重置参数函数以初始化权重和偏置
        self.reset_parameters()

    def reset_parameters(self):
        # 计算标准差，用于参数的均匀分布初始化
        # 使用权重矩阵的列数的平方根的倒数作为标准差
        stdv = 1. / math.sqrt(self.weight.size(1))

        # 将权重矩阵初始化为在[-stdv, stdv]范围内的均匀分布
        # 这是He初始化的一种形式，用于在训练开始时提供一个好的参数起始点
        self.weight.data.uniform_(-stdv, stdv)

        # 如果存在偏置项（bias），则也将偏置项初始化为[-stdv, stdv]范围内的均匀分布
        # 这也是一种常见的初始化方法，可以帮助网络在训练初期避免被卡在饱和区
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向传播函数
    def forward(self, x, G):
        # 将输入特征x与权重矩阵相乘
        x = x.matmul(self.weight)
        # 如果存在偏置项，则将偏置加到x上
        if self.bias is not None:
            x = x + self.bias
            # 将加权后的节点特征与图的邻接矩阵G相乘，实现特征的聚合
        x = G.matmul(x)
        return x

    # 定义类的字符串表示，用于打印类实例时显示更多信息
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 定义一个图卷积网络模型，用于学习图上的节点表示
class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout=0.5):
        super(HGCN, self).__init__()  # 调用基类的构造函数
        # 初始化模型中的dropout比例
        self.dropout = dropout

        # 初始化第一个图卷积层，将输入维度映射到第一个隐藏层的维度
        self.hgnn1 = HGNN_conv(in_dim, hidden_list[0])

    # 定义前向传播函数
    def forward(self, x, G):
        # 将输入特征x通过第一个图卷积层，得到嵌入表示
        x_embed = self.hgnn1(x, G)

        # 应用LeakyReLU激活函数，参数0.25表示负斜率
        # LeakyReLU允许负值有小的梯度，这有助于避免神经元死亡
        x_embed_1 = F.leaky_relu(x_embed, 0.25)

        # 返回经过激活函数处理后的嵌入表示
        return x_embed_1

# 对比学习框架下的图卷积网络模型
class CL_HGCN(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, alpha = 0.5):
        super(CL_HGCN, self).__init__()# 调用基类的构造函数
        # 初始化两个相同的图卷积网络，用于提取正样本对的特征
        self.hgcn1 = HGCN(in_size, hid_list)# 图卷积网络的第一次应用
        self.hgcn2 = HGCN(in_size, hid_list) # 图卷积网络的第二次应用
        # 初始化两个全连接层，用于将图卷积网络的输出投影到低维空间
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)# 第一个全连接层
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])# 第二个全连接层，映射回原始维度
        # 设置温度参数tau，用于控制相似度的缩放
        self.tau = 0.5
        # 设置alpha参数，用于平衡正样本对和负样本对的损失
        self.alpha = alpha

    # 前向传播函数，计算对比损失
    def forward(self, x1, adj1, x2, adj2):
        # 通过第一个HGCN模型提取的两个视图的特征
        z1 = self.hgcn1(x1, adj1)# 第一个视图的特征
        h1 = self.projection(z1) # 第一个视图的投影特征

        z2 = self.hgcn2(x2, adj2)# 第二个视图的特征
        h2 = self.projection(z2)# 第二个视图的投影特征
        # 计算对比损失，结合了两个方向的相似度
        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)
        # 返回提取的特征和计算的对比损失
        return z1, z2, loss

    # 投影函数，将特征映射到低维空间
    def projection(self, z):
        # 通过第一个全连接层，并应用Elu激活函数
        z = F.elu(self.fc1(z))
        # 通过第二个全连接层映射回原始维度
        return self.fc2(z)

    # 计算归一化的相似度矩阵
    def norm_sim(self, z1, z2):
        # 对两个特征集进行归一化
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # 计算归一化特征之间的相似度矩阵
        return torch.mm(z1, z2.t())

    # 计算相似度损失
    def sim(self, z1, z2):
        # 定义一个函数，用于应用温度缩放的指数函数
        f = lambda x: torch.exp(x / self.tau)
        # 计算自身相似度矩阵
        refl_sim = f(self.norm_sim(z1, z1))
        # 计算不同视图之间的相似度矩阵
        between_sim = f(self.norm_sim(z1, z2))
        # 计算对比损失，使用对数函数和温度缩放
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        # 对损失进行求和和平均
        loss = loss.sum(dim=-1).mean()
        return loss

# 引入注意力机制的图卷积网络模型
class HGCN_Attention_mechanism(nn.Module):
    def __init__(self):
        super(HGCN_Attention_mechanism,self).__init__()# 调用基类的构造函数
        # 初始化注意力机制参数
        self.hiddim = 64
        # 定义两个全连接层，用于处理输入特征和生成注意力分数
        self.fc_x1 = nn.Linear(in_features=2, out_features=self.hiddim)# 第一个全连接层
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=2)# 第二个全连接层
        # 定义Sigmoid激活函数，用于生成(0, 1)之间的注意力权重
        self.sigmoidx = nn.Sigmoid()

    # 前向传播函数，应用注意力机制
    def forward(self,input_list):
        # 将输入列表中的两个张量沿着第二维（列）拼接起来
        XM = torch.cat((input_list[0], input_list[1]), 1).t()
        # 重塑张量形状，以适应后续的二维平均池化操作
        XM = XM.view(1, 1 * 2, input_list[0].shape[1], -1)
        # 使用二维平均池化来聚合通道和时间步的信息
        globalAvgPool_x = nn.AvgPool2d((input_list[0].shape[1], input_list[0].shape[0]), (1, 1))
        x_channel_attenttion = globalAvgPool_x(XM)
        # 重塑平均池化后的张量，以匹配全连接层的输入要求
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        # 通过第一个全连接层
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)
        # 应用ReLU激活函数
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        # 通过第二个全连接层
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)
        # 应用Sigmoid激活函数，生成注意力分数
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        # 重塑注意力分数张量，以与原始特征张量相乘
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        # 将注意力分数与原始特征张量相乘，实现加权
        XM_channel_attention = x_channel_attenttion * XM
        # 应用ReLU激活函数
        XM_channel_attention = torch.relu(XM_channel_attention)
        # 返回第一个元素，即处理后的特征张量
        return XM_channel_attention[0]

# 综合了图卷积网络、对比学习和Transformer的模型
class HGCLAMIR(nn.Module):
    def __init__(self, mi_num, dis_num, hidd_list, num_proj_hidden, hyperpm):
        super(HGCLAMIR, self).__init__()# 调用基类的构造函数
        # 初始化模型参数，包括对比学习模块、注意力机制和Transformer编码器
        self.CL_HGCN_mi = CL_HGCN(mi_num + dis_num, hidd_list,num_proj_hidden)
        self.CL_HGCN_dis = CL_HGCN(dis_num + mi_num, hidd_list,num_proj_hidden)

        # 初始化piRNA和疾病数据的注意力机制模块
        self.AM_mi = HGCN_Attention_mechanism()
        self.AM_dis = HGCN_Attention_mechanism()
        # 初始化piRNA和疾病数据的Transformer编码器
        self.Transformer_mi = TransformerEncoder([hidd_list[-1],hidd_list[-1]], hyperpm)
        self.Transformer_dis = TransformerEncoder([hidd_list[-1],hidd_list[-1]], hyperpm)

        # 初始化piRNA和疾病数据的下游任务的线性变换层
        self.linear_x_1 = nn.Linear(hyperpm.n_head*hyperpm.n_hidden*hyperpm.nmodal, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, 64)

        self.linear_y_1 = nn.Linear(hyperpm.n_head*hyperpm.n_hidden*hyperpm.nmodal, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, 64)

    # 前向传播函数，整合各个组件进行联合学习
    def forward(self, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
        # 将piRNA和疾病数据作为图嵌入的输入
        mi_embedded = concat_mi_tensor
        dis_embedded = concat_dis_tensor

        # 通过对比学习模型获取piRNA特征和损失
        mi_feature1, mi_feature2, mi_cl_loss = self.CL_HGCN_mi(mi_embedded, G_mi_Kn, mi_embedded, G_mi_Km)
        # 应用注意力机制获取加权的piRNA特征
        mi_feature_att = self.AM_mi([mi_feature1,mi_feature2])
        # 转置注意力机制的输出以匹配Transformer的输入格式
        mi_feature_att1 = mi_feature_att[0].t()
        mi_feature_att2 = mi_feature_att[1].t()
        # 拼接加权的piRNA特征
        mi_concat_feature = torch.cat([mi_feature_att1, mi_feature_att2], dim=1)
        mi_feature = self.Transformer_mi(mi_concat_feature)

        # 通过Transformer编码器获取最终的miRNA特征
        dis_feature1, dis_feature2, dis_cl_loss = self.CL_HGCN_dis(dis_embedded, G_dis_Kn, dis_embedded, G_dis_Km)
        # 通过对比学习模型获取疾病特征和损失
        dis_feature_att = self.AM_dis([dis_feature1,dis_feature2])
        # 转置注意力机制的输出以匹配Transformer的输入格式
        dis_feature_att1 = dis_feature_att[0].t()
        dis_feature_att2 = dis_feature_att[1].t()
        # 拼接加权的疾病特征
        dis_concat_feature = torch.cat([dis_feature_att1, dis_feature_att2], dim=1)
        # 通过Transformer编码器获取最终的疾病特征
        dis_feature = self.Transformer_dis(dis_concat_feature)
        # 通过线性层和ReLU激活函数获取miRNA数据的下游任务分数
        x1 = torch.relu(self.linear_x_1(mi_feature))
        x2 = torch.relu(self.linear_x_2(x1))
        x = torch.relu(self.linear_x_3(x2))
        # 通过线性层和ReLU激活函数获取疾病数据的下游任务分数
        y1 = torch.relu(self.linear_y_1(dis_feature))
        y2 = torch.relu(self.linear_y_2(y1))
        y = torch.relu(self.linear_y_3(y2))

        # 计算得分矩阵，通常是两个特征向量的点积结果
        score = x.mm(y.t())
        # filename = 'HGCLAMIR-master/score_matrix.txt'
        # # 确保得分矩阵是一个二维数组
        # if score.dim() == 1:
        #     score = score.unsqueeze(0)
        # # 将得分矩阵转换为numpy数组
        # score_numpy = score.cpu().detach().numpy()
        # # 将numpy数组保存到txt文件
        # np.savetxt(filename, score_numpy)

        # 返回预测分数、miRNA对比损失和疾病对比损失
        return score, mi_cl_loss, dis_cl_loss



