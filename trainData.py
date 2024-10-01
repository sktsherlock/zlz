from __future__ import division


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset# 存储传入的数据集
        self.nums = opt.validation# 存储验证集的大小

    def __getitem__(self, index):
        """
        根据索引index获取数据集中的一项。

        参数:
        index : 要获取的数据项的索引。

        返回:
        一个包含数据集项的元组，包括：
        - self.data_set['ID']：数据集的ID。
        - self.data_set['IM']：数据集的图像数据。
        - self.data_set['md'][index]['train']：第index个数据集的标记数据的训练部分。
        - self.data_set['md'][index]['test']：第index个数据集的标记数据的测试部分。
        - self.data_set['md_p']：标记数据的预测。
        - self.data_set['md_true']：标记数据的真实值。
        - self.data_set['independent'][0]['train']：独立数据的训练部分。
        - self.data_set['independent'][0]['test']：独立数据的测试部分。
        """
        return (self.data_set['ID'], self.data_set['IM'],
                self.data_set['md'][index]['train'], self.data_set['md'][index]['test'],
                self.data_set['md_p'], self.data_set['md_true'],
                self.data_set['independent'][0]['train'],self.data_set['independent'][0]['test'])

    def __len__(self):
        """
        返回数据集中验证集的大小。

        这通常用于指示数据加载器可以加载多少项。
        """
        return self.nums



