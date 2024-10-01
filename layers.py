
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

# 设置设备，优先使用GPU，如果没有GPU，则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 注意力机制模块
class Attention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()  # 调用基类的构造函数
        self.temperature = temperature  # 温度参数，用于缩放注意力分数
        self.dropout = nn.Dropout(attn_dropout)  # dropout层，防止过拟合

    def forward(self, q, k, v, mask=None):
        # q: 查询 Queries (bsz, seq_len, dim)
        # k: 键 Keys (bsz, seq_len, dim)
        # v: 值 Values (bsz, seq_len, dim)
        # mask: 掩码，用于屏蔽某些位置的注意力分数
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # 计算注意力分数
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # 使用掩码屏蔽某些位置
        attn = attn / abs(attn.min())  # 缩放注意力分数
        attn = self.dropout(F.softmax(attn, dim=-1))  # 应用softmax和dropout
        output = torch.matmul(attn, v)  # 计算加权的值
        return output, attn, v  # 返回输出和注意力分数

# 前馈网络层
class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # 第一个线性变换
        self.w_2 = nn.Linear(d_hid, d_in)  # 第二个线性变换
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)  # 层归一化
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, x):
        residual = x  # 保存原始输入以进行残差连接
        x = self.w_2(F.gelu(self.w_1(x)))  # 应用前馈网络
        x = self.dropout(x)  # 应用dropout
        x += residual  # 残差连接
        x = self.layer_norm(x)  # 应用层归一化
        return x

# 可变长度输入层
class VariLengthInputLayer(nn.Module):
    def __init__(self, input_data_dims, d_k, d_v, n_head, dropout):
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head  # 注意力头的数量
        self.dims = input_data_dims  # 输入数据的维度
        self.d_k = d_k  # 键的维度
        self.d_v = d_v  # 值的维度
        self.w_qs = []  # 查询权重
        self.w_ks = []  # 键权重
        self.w_vs = []  # 值权重
        # 对于每种输入数据维度，创建对应的查询、键、值权重
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_k = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_v = nn.Linear(dim, n_head * d_v, bias=False)
            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module('linear_q_%d_%d' % (dim, i), self.w_q)
            self.add_module('linear_k_%d_%d' % (dim, i), self.w_k)
            self.add_module('linear_v_%d_%d' % (dim, i), self.w_v)
        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)  # 注意力机制
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)  # 线性变换
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)  # 层归一化

    def forward(self, input_data, mask=None):
        # 初始化临时维度计数器和批量大小
        temp_dim = 0
        bs = input_data.size(0)
        modal_num = len(self.dims)
        # 初始化查询、键、值的张量
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).to(device)

        # 为每种模态生成查询、键、值
        for i in range(modal_num):
            w_q = self.w_qs[i]
            w_k = self.w_ks[i]
            w_v = self.w_vs[i]

            # 根据模态维度提取数据子集
            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]

            # 应用对应的权重矩阵
            q[:, i, :] = w_q(data)
            k[:, i, :] = w_k(data)
            v[:, i, :] = w_v(data)

        # 调整查询、键、值的视角和形状
        q = q.view(bs, modal_num, self.n_head, self.d_k)
        k = k.view(bs, modal_num, self.n_head, self.d_k)
        v = v.view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 应用注意力机制
        q, attn, residual = self.attention(q, k, v)
        # 调整查询的形状以匹配后续操作
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        # 应用dropout和线性变换
        q = self.dropout(self.fc(q))
        # 加上残差连接
        q += residual
        # 应用层归一化
        q = self.layer_norm(q)
        # 返回处理后的数据和注意力权重
        return q, attn


# 编码层，用于Transformer模型中
class EncodeLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dropout):
        super(EncodeLayer, self).__init__()
        self.n_head = n_head  # 注意力头的数量
        self.d_k = d_k  # 键的维度
        self.d_v = d_v  # 值的维度
        # 创建查询、键、值的线性变换层
        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)  # 输出线性变换层
        self.attention = Attention(temperature=d_k ** 0.5)  # 注意力机制
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化

    def forward(self, q, k, v, modal_num, mask=None):
        # 记录原始查询数据，用于残差连接
        bs = q.size(0)
        residual = q
        # 通过权重矩阵生成查询、键、值，并调整视角和形状
        q = self.w_q(q).view(bs, modal_num, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, modal_num, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # 应用注意力机制，传入掩码以避免在未来的值上进行计算
        q, attn, _ = self.attention(q, k, v, mask=mask)
        # 调整查询的形状以匹配后续操作，并进行dropout处理
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual        # 加上残差连接
        q = self.layer_norm(q)        # 应用层归一化
        # 返回处理后的数据和注意力权重
        return q, attn


