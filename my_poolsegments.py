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
        """if mode == 'sum':
            self._reduce = torch.segment_sum
        elif mode == 'mean':
            self._reduce = torch.segment_mean"""
        super(PoolSegments, self).__init__()

    def forward(self, x, segs):
        # print(x)
        # print(segs)
        # y = self._reduce(x, segs)
        # return y
        pass
