import argparse

def parameter_parser():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Run Model.")
    # 添加命令行参数
    # --data_path: 指定数据文件的路径，默认为'../data_piRDisease'
    parser.add_argument('--data_path', type=str, default='data_miRDisease',
                        help='the number of miRANs.')
    # --validation: 指定验证集的大小，默认为10
    parser.add_argument('--validation', type=int, default=5,
                        help='the number of miRANs.')
    # --epoch: 指定训练的轮数，默认为650
    parser.add_argument('--epoch', type=int, default=650,
                        help='the number of epoch.')
    # --pi_num: 数据集miRNA的数量
    parser.add_argument('--pi_num', type=int, default=4349,
                        help='the number of miRANs.')
    # --dis_num: 指定疾病的数量
    parser.add_argument('--dis_num', type=int, default=21,
                        help='the number of diseases.')
    # --alpha: 指定alpha的大小，默认为0.11
    parser.add_argument('--alpha', type=int, default=0.15,
                        help='the size of alpha.')
    # --dropout: 指定dropout率，默认为0.1
    parser.add_argument('--dropout', type=float, default=0.7,
                        help='Dropout rate (1 - keep probability).')
    # --nlayer: 指定模型的层数，默认为2
    parser.add_argument('--nlayer', type=int, default=2,
                        help='Number of layers.')
    # --n_hidden: 指定每模态的隐藏单元数，默认为20
    parser.add_argument('--n_hidden', type=int, default=20,
                        help='Number of hidden units per modal.')
    # --n_head: 指定注意力机制中的头数，默认为5
    parser.add_argument('--n_head', type=int, default=5,
                        help='Number of attention head.')
    # --nmodal: 指定视图的数量，默认为2
    parser.add_argument('--nmodal', type=int, default=2,
                        help='Number of views.')
    # 解析命令行参数并返回
    return parser.parse_args()