import torch
import torch.nn as nn
# import tensorflow as tf
from torch.functional import F

class EuclideanDistances(nn.Module):
    def __init__(self, name=None, dtype=torch.float32):
        super(EuclideanDistances, self).__init__()
        self.dtype = dtype

    def forward(self, r:torch.Tensor,  # torch.Size([batch*n, 3])
                offsets:torch.Tensor,
                idx_i:torch.Tensor,
                idx_j:torch.Tensor):
        if not isinstance(r, torch.FloatType):
            # make sure the input is tensor-like
            print("chaning the type of tensor...")
            r = r.type(self.dtype)
            print("done!")
        # idx_i:当前原子下标
        # idx_j:需要遍历的其他原子下标

        ri = torch.index_select(r, dim=0, index=idx_i)

        rj = torch.index_select(r, dim=0, index=idx_j) + offsets

        rij = ri - rj

        # 计算原子之间的距离
        dij2 = torch.sum(rij ** 2, dim=-1, keepdim=True)
        dij = torch.sqrt(dij2)  # (batch*n*(n-1), 1)


        return dij