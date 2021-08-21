import torch
import torch.nn as nn
from my_utils import shape
from my_initializer import *

class Dense(nn.Module):
    def __init__(self,
                 fan_in: int,
                 fan_out: int,
                 use_bias=True, activation=None,
                 trainable=True, name=None, dtype=torch.float32,
                 gpu=False):
        super(Dense, self).__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.use_bias = use_bias
        self.activation = activation
        self.trainable = trainable
        # shape = (self._fan_in, self._fan_out)
        self.dtype = dtype
        self.gpu = gpu
        self.W = glorot_uniform(self.fan_in, self.fan_out)
        self.b = torch.zeros((self.fan_out,), dtype=self.dtype)
        
        if self.gpu:
            self.W = self.W.cuda()
            self.b = self.b.cuda()
        
        self.W = nn.Parameter(self.W)
        self.b = nn.Parameter(self.b)
        self.register_parameter("{0}_{1}".format(name, "W"), self.W)
        self.register_parameter("{0}_{1}".format(name, "b"), self.b)

    def forward(self, x: torch.Tensor):
        # print(x)
        if not isinstance(x, torch.FloatType):
            # make sure the input is tensor-like
            # print("chaning the type of tensor...")
            x = x.type(self.dtype)
            # print("done!")

        x_shape = shape(x)  # (n,fan_in)
        # print(x_shape)
        # print(self._fan_in)
        ndims = len(x_shape)  # 2
        # reshape for broadcasting
        if not x_shape[-1] == self.fan_in:
            print(x_shape[-1])
            print(self.fan_in)
        x_r = torch.reshape(x, (-1, self.fan_in))
        # Weight and tensor product
        # print("x_r:", x_r.device)
        # print("self.W:", self.W.device)
        y = torch.matmul(x_r, self.W)
        # Whether to use bias
        if self.use_bias:
            y += self.b
        # activation
        if self.activation:
            y = self.activation(y)
        new_shape = (x_shape[0], self.fan_out)

        y = torch.reshape(y, new_shape)

        return y


