# import tensorflow as tf
import torch.nn as nn
"""  
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_sum(c, tf.constant([0, 0, 1]))
# ==> [[5, 5, 5, 5],
#      [5, 6, 7, 8]]
"""
import torch

class PoolSegments(nn.Module):
    def __init__(self, mode='sum', name=None):
        if mode == 'sum':
            self._reduce = torch.index_add
        elif mode == 'mean':
            pass
            # self._reduce = torch.segment_mean
        super(PoolSegments, self).__init__()

    def forward(self, x, segs, segs_pool):
        # print("x:", x.shape, x)
        # print("segs:", segs.shape, segs)
        # print("segs_pool:", segs_pool.shape, segs_pool)
        zeros = torch.zeros(size=(x.shape))
        # torch.index_add(input=zeros, dim=0, index=segs, source=x)  # out-of-place
        y = self._reduce(input=zeros, dim=0, index=segs, source=x)
        y = torch.index_select(input=y, dim=0, index=segs_pool)
        # print(segs)
        # print(y[:,:2])
        return y
