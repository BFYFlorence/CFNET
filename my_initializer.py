import torch
import numpy as np



def glorot_uniform(fan_in:int, fan_out:int):
    r = np.sqrt(6. / (fan_in + fan_out))

    return torch.nn.init.uniform_(torch.empty(fan_in, fan_out), a=-r, b=r)