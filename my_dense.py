import torch
import torch.nn as nn
from my_utils import shape

class Dense(nn.Module):
    def __init__(self,
                 fan_in:int,
                 fan_out:int,
                 use_bias=True, activation=None,
                 trainable=True, name=None, dtype=torch.float32):
        super(Dense, self).__init__()
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._use_bias = use_bias
        self.activation = activation
        self._trainable = trainable
        # shape = (self._fan_in, self._fan_out)
        self._dtype = dtype
        self.W = torch.ones((self._fan_in, self._fan_out), dtype=self._dtype)
        self.b = torch.zeros((self._fan_out,), dtype=self._dtype)

    def forward(self, x:torch.Tensor):
        # print(x)
        if not isinstance(x, torch.FloatType):
            # make sure the input is tensor-like
            print("chaning the type of tensor...")
            x = x.type(self._dtype)
            print("done!")

        x_shape = shape(x)  # (n,fan_in)
        # print(x_shape)
        # print(self._fan_in)
        ndims = len(x_shape)  # 2
        # reshape for broadcasting
        if not x_shape[-1] == self._fan_in:
            print(x_shape[-1])
            print(self._fan_in)
        x_r = torch.reshape(x, (-1, self._fan_in))
        # 权重与张量乘积
        y = torch.matmul(x_r, self.W)
        # 是否使用偏置
        if self._use_bias:
            y += self.b
        # 激活
        if self.activation:
            y = self.activation(y)
        new_shape = (x_shape[0], self._fan_out)

        y = torch.reshape(y, new_shape)

        return y


