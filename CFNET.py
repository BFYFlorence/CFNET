import torch
import torch.nn as nn
import numpy as np
from my_dense import Dense
from my_activation import shifted_softplus

from my_poolsegments import PoolSegments

"""
tensor.FloatTensor
tensor.LongTensor
tensor.ByteTensor
tensor.CharTensor
tensor.ShortTensor
tensor.IntTensor
torch.LongTensor
其中torch.Tensor是默认的tensor.FloatTensor的简称。
"""

"""
atom_refs=None,
basis=128,
batch_size=32,
cutoff=20.0,
filters=128,
interactions=6,

energy='total_energy',
eweight=0.1,
filter_pool_mode='sum',
fit_energy=True,
fit_force=True,
forces='atomic_forces',
intensive=False,

keep_prob=1.0,
lr=0.0001,
max_steps=5000000,
name='',
normalized_filters=False,
output_dir='./results/c20',
shared_interactions=False,
train_data='data/c20_splits/split20k_0/train.db',
val_data='data/c20_splits/split20k_0/validation.db',
valbatch=500,
valint=5000,
valsize=1000
"""

class CFConv(nn.Module):
    # Continuous-filter convolution layer
    def __init__(self, fan_in, fan_out, n_filters, pool_mode='sum',
                 activation=None, name=None):
        super(CFConv, self).__init__()
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._n_filters = n_filters  # 128
        self.activation = activation
        self.pool_mode = pool_mode
        self.in2fac = Dense(self._fan_in, self._n_filters, use_bias=False,
                            name='in2fac')
        self.fac2out = Dense(self._n_filters, self._fan_out, use_bias=True,
                             activation=self.activation,
                             name='fac2out')
        self.pool = PoolSegments(mode=self.pool_mode)

    def forward(self, x, w_ij, seg_i, idx_j, seg_i_sum):
        '''
            :param x (num_atoms, num_feats): input
            :param w (num_interactions, num_filters): filters
            :param seg_i (num_interactions,): segments of atom i
            :param idx_j: (num_interactions,): indices of atom j
            :return: convolution x * w
        '''
        # to filter-space

        f = self.in2fac(x)
        # filter-wise convolution
        f = torch.index_select(f, dim=0, index=idx_j)
        # print("idx_j:", idx_j)
        print("f:", f.shape, f)

        wf = w_ij * f
        print("wf:", wf.shape, wf)
        print("seg_i:", seg_i)
        # print(wf[:,:2])
        conv = self.pool(wf, seg_i, seg_i_sum)
        # print(conv[0][:2])
        # print(conv[1][:2])

        # to output-space
        c = self.fac2out(conv)
        print("c:", c.shape, c)
        return c

class CFnetFilter(nn.Module):
    def __init__(self, input_size, filters_num, pool_mode='sum', name=None):
        super(CFnetFilter, self).__init__()
        self.input_size = input_size
        self.filters_num = filters_num
        self.pool_mode = pool_mode
        self.dense1 = Dense(input_size, filters_num, activation=shifted_softplus)
        self.dense2 = Dense(filters_num, filters_num, activation=shifted_softplus)
        self.pooling = PoolSegments(self.pool_mode)

    def forward(self, dijk, seg_j, ratio_j=1.):
        h = self.dense1(dijk)
        w_ijk = self.dense2(h)
        w_ij = self.pooling(w_ijk, seg_j, seg_j)

        return w_ij

class CFNetInteractionBlock(nn.Module):
    def __init__(self, n_in, n_basis, n_filters, pool_mode='sum',
                 name=None):
        self.n_in = n_in
        self.n_basis = n_basis  # 128
        self.n_filters = n_filters  # 128
        self.pool_mode = pool_mode
        super(CFNetInteractionBlock, self).__init__()
        self.filternet = CFnetFilter(self.n_in, self.n_filters,
                                      pool_mode=self.pool_mode)

        self.cfconv = CFConv(self.n_basis, self.n_basis, self.n_filters,
                             activation=shifted_softplus)

        self.dense = Dense(self.n_basis, self.n_basis)

    def forward(self, x, dijk, idx_j, seg_i, seg_j, seg_i_sum,ratio_j=1.):
        w_ij = self.filternet(dijk, seg_j, ratio_j)
        # print("w_ij:",w_ij.shape,w_ij[:,:2])
        c = self.cfconv(x, w_ij, seg_i, idx_j, seg_i_sum)
        v = self.dense(c)

        y = x + v
        return y, v


class CFnet(nn.Module):
    def __init__(self, n_interactions, n_basis, n_filters, cutoff,
                 mean_per_atom=np.zeros((1,), dtype=np.float32),
                 std_per_atom=np.ones((1,), dtype=np.float32),
                 gap=0.1, atomref=None, intensive=False,
                 filter_pool_mode='sum',
                 return_features=False,
                 shared_interactions=False,
                 atomization_energy=False,
                 n_embeddings=100):
        self.n_interactions = n_interactions
        self.n_basis = n_basis
        self.n_filters = n_filters
        self.n_embeddings = n_embeddings
        self.cutoff = cutoff
        self.atomization_energy = atomization_energy
        self.shared_interactions = shared_interactions
        self.return_features = return_features
        self.intensive = intensive
        self.filter_pool_mode = filter_pool_mode
        self.atomref = atomref
        self.gap = gap

        self.mean_per_atom = mean_per_atom
        self.std_per_atom = std_per_atom
        super(CFnet, self).__init__()

    def forward(self):
        pass
