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
        # idx_i: Current atomic index
        # idx_j: Other atomic subscripts that need to be traversed

        ri = torch.index_select(r, dim=0, index=idx_i)

        rj = torch.index_select(r, dim=0, index=idx_j) + offsets

        rij = ri - rj

        # Calculate the distance between atoms
        dij2 = torch.sum(rij ** 2, dim=-1, keepdim=True)
        dij = torch.sqrt(dij2)  # (batch*n*(n-1), 1)


        return dij