from torchvision import transforms, utils
from torch.utils.data import Dataset
from autoEncoder.data.getData import getData
from PIL import Image
import numpy as np
import torch.optim as optim
import os

root = os.getcwd() + '/autoEncoder/data/'  # 数据集的地址



class TrainDataset(Dataset):
    # 创建自己的类： TrainDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, name, relOrent):
        super(TrainDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        data = getData(os.path.join(root,name), relOrent)
        #!! 对数据集进行归一化处理(可以尝试不同的归一化方法）
        min = data.min()
        max = data.max()
        data = (data-min)/(max-min)
        self.data = data
        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        return self.data[index]
        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)