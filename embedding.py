import torch
from my_dense import Dense
# import tensorflow as tf
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_embeddings,
                 dim,  # 64
                 # embedding_init=None,
                 trainable=True,
                 name=None,
                 dtype=torch.float32,
                 gpu=False):
        super(Embedding, self).__init__()
        self.n_embeddings = n_embeddings
        self.dim = dim
        # self.embedding_init = embedding_init
        self.trainable = trainable
        self.dtype = dtype
        self.gpu = gpu
        self.embeddings = Dense(self.n_embeddings, self.dim, use_bias=False,
                                trainable=self.trainable,
                                name="{0}_embeddings".format(name))
        """if self.gpu:
            self.embeddings = self.embeddings.cuda()"""
        
    def forward(self, indices:torch.Tensor):
        I = torch.eye(self.n_embeddings, dtype=self.dtype)
        if self.gpu:
            I = I.cuda()
        # 1. 0. 0. 0. 0.
        # 0. 1. 0. 0. 0.
        # 0. 0. 1. 0. 0.
        # 0. 0. 0. 1. 0.
        # 0. 0. 0. 0. 1.  ...I:The range of one-hot encoding to be selected(n,n)，as long as n>m is satisfied
        ind = torch.index_select(I, dim=0, index=indices)
        # print(ind)  # one-hot
        y = self.embeddings(ind)  # Initialization will all be 1, this will be optimized later
        # print(y)
        return y
