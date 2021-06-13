import torch
import torch.nn as nn
# import tensorflow as tf
from torch.functional import F

class EuclideanDistances(nn.Module):
    def __init__(self, name=None, dtype=torch.float32):
        super(EuclideanDistances, self).__init__()
        self.dtype = dtype

    def forward(self, r:torch.Tensor,  # torch.Size([n, 3])
                offsets:torch.Tensor,
                idx_ik:torch.Tensor,
                idx_jk:torch.Tensor):
        if not isinstance(r, torch.FloatType):
            # make sure the input is tensor-like
            print("chaning the type of tensor...")
            r = r.type(self.dtype)
            print("done!")
        # idx_ik:当前原子下标
        # idx_jk:需要遍历的其他原子下标

        ri = torch.index_select(r, dim=0, index=idx_ik)  # broadcast
        # print("r:", r, r.shape)
        # print("idx_jk:", idx_jk, idx_jk.shape)

        rj = torch.index_select(r, dim=0, index=idx_jk) + offsets

        rij = ri - rj

        # 计算原子之间的距离
        dij2 = torch.sum(rij ** 2, dim=-1, keepdim=True)
        dij = torch.sqrt(F.relu(dij2))  # (n*(n-1), 1)


        return dij