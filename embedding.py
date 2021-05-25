import torch
from my_dense import Dense
import numpy as np
# import tensorflow as tf
from my_utils import molecules

class Embedding:
    def __init__(self, n_embeddings,
                 dim,  # 128
                 # embedding_init=None,
                 trainable=True,
                 name=None,
                 dtype=torch.float32):
        super(Embedding, self).__init__()
        self.n_embeddings = n_embeddings
        self.dim = dim
        # self.embedding_init = embedding_init
        self.trainable = trainable
        self.dtype = dtype

        self.embeddings = Dense(self.n_embeddings, self.dim, use_bias=False,
                                trainable=self.trainable,
                                name='embeddings')

    def forward(self, indices:torch.Tensor):
        I = torch.eye(self.n_embeddings, dtype=self.dtype)

        # 1. 0. 0. 0. 0.
        # 0. 1. 0. 0. 0.
        # 0. 0. 1. 0. 0.
        # 0. 0. 0. 1. 0.
        # 0. 0. 0. 0. 1.  ...I
        ind = torch.index_select(I, dim=0, index=indices)
        # print(ind)  # one-hot
        y = self.embeddings(ind)  # 初始化会全为1，之后再优化
        # print(y)
        return y
