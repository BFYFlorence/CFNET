import numpy as np
import torch
import torch.nn as nn
from my_utils import shape

class RBF(nn.Module):
    def __init__(self, low, high, gap,
                 dim=1,
                 name=None, dtype=torch.float32):
        super(RBF, self).__init__()
        self.low = low
        self.high = high
        self.gap = gap  # 决定rbf的间隔
        self.dim = dim  # 维度扩充
        xrange = high - low
        bins = int(np.ceil(xrange / gap))  # 计算在此间隔下，允许划分的组数
        self.centers = np.linspace(low, high, bins)  # (bins,)
        self.centers = np.expand_dims(self.centers, -1)  # (bins,1)

        self.n_centers = len(self.centers)  # bins
        self._dtype = dtype
        self.fan_out = self.dim * self.n_centers

    def forward(self,
                d:torch.Tensor  # (1,)
                ):
        if not isinstance(d, torch.FloatType):
            # make sure the input is tensor-like
            print("chaning the type of tensor...")
            d = d.type(self._dtype)
            print("done!")

        d_shape = shape(d)  # torch.Size([1])

        centers = self.centers.reshape((-1,)).astype(np.float32)

        d -= centers  # 计算偏离每个中心的距离
        rbf = torch.exp(-(d ** 2) / self.gap)  # 计算径向基函数值
        rbf = rbf.reshape((d_shape[0], self.fan_out))
        return rbf

# rbf = RBF(0.,30.,0.1)
# print(rbf.fan_out)
# d = np.array([1.5])
# rbf.forward(torch.tensor(d))

