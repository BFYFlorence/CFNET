import numpy
import numpy as np
import torch
import torch.nn as nn
from my_utils import shape

class RBF(nn.Module):
    def __init__(self, low, high, gap,
                 dim=torch.tensor(1, dtype=torch.int32),
                 name=None, dtype=torch.float32,
                 gpu=False):
        super(RBF, self).__init__()
        self.dtype = dtype
        self.gpu = gpu
        self.low = low
        self.high = high
        self.gap = gap  # 决定rbf的间隔
        self.dim = dim.cuda() if self.gpu else dim  # 维度扩充
        xrange = high - low
        bins = torch.ceil(xrange / gap).int()  # 计算在此间隔下，允许划分的组数
        self.centers = torch.linspace(low, high, bins)  # (bins,)
        self.centers = torch.tensor(np.expand_dims(self.centers, -1), dtype=self.dtype)  # (bins,1)
        self.n_centers = len(self.centers)  # bins
        self.fan_out = self.dim * self.n_centers
        
        if self.gpu:
            self.centers = self.centers.cuda()
            # self.gap = self.gap.cuda()
            self.fan_out = self.fan_out.cuda()
            
    def forward(self,
                d:torch.Tensor  # (1,)
                ):
        if not isinstance(d, torch.FloatType):
            # make sure the input is tensor-like
            print("chaning the type of tensor...")
            d = d.type(self.dtype)
            print("done!")

        d_shape = shape(d)  # (batch*n*(n-1), 1)

        centers = self.centers.reshape((-1,))

        d = d - centers  # 计算偏离每个中心的距离；不可以写成d-=centers,这样会报形状错误
        rbf = torch.exp(-(d ** 2) / self.gap)  # 计算径向基函数值
        rbf = rbf.reshape((d_shape[0], self.fan_out))

        return rbf

# rbf = RBF(0.,30.,0.1)
# print(rbf.fan_out)
# d = np.array([1.5])
# rbf.forward(torch.tensor(d))