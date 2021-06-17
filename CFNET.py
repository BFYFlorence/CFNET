from my_rbf import RBF
from distances import EuclideanDistances
import torch
from embedding import Embedding
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
General semantics:
Each tensor has at least one dimension.

When iterating over the dimension sizes, starting at the trailing dimension, 
the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

If two tensors x, y are “broadcastable”, the resulting tensor size is calculated as follows:

If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.
"""

# 只有将自定义参数登记入模型中，使用optimizer清洗梯度时才能够正确执行，否则的话未登记的参数梯度会一直积累叠加
# 如果牵涉到数学运算，不论是四则运算还是高级运算，都要保证数据位于同一个设备上
# pytorch有一些函数，并不是只要输入都是cuda上，输出就一定在cuda上，例如torch.linspace

class CFConv(nn.Module):
    # Continuous-filter convolution layer
    def __init__(self, fan_in, fan_out, nFM, pool_mode='sum',
                 activation=None, name=None, gpu=False):
        super(CFConv, self).__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.gpu = gpu
        self.nFM = nFM  # 128
        self.activation = activation
        self.pool_mode = pool_mode
        self.in2fac = Dense(self.fan_in, self.nFM, use_bias=False,
                            name='{0}_in2fac'.format(name), gpu=self.gpu)
        self.fac2out = Dense(self.nFM, self.fan_out, use_bias=True,
                             activation=self.activation,
                             name='{0}_fac2out'.format(name), gpu=self.gpu)
        self.pool = PoolSegments(mode=self.pool_mode, gpu=self.gpu)

        if self.gpu:
            self.in2fac = self.in2fac.cuda()
            self.fac2out = self.fac2out.cuda()
            self.pool = self.pool.cuda()

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
        # print("f:", f.shape, f)

        wf = w_ij * f
        # print("wf:", wf.shape, wf)
        # print("seg_i:", seg_i)
        # print(wf[:,:2])
        conv = self.pool(wf, seg_i, seg_i_sum)
        # print(conv[0][:2])
        # print(conv[1][:2])

        # to output-space
        c = self.fac2out(conv)
        # print("c:", c.shape, c)
        return c

class CFnetFilter(nn.Module):
    def __init__(self, input_size, filters_num, pool_mode='sum', name=None, gpu=False):
        super(CFnetFilter, self).__init__()
        self.input_size = input_size
        self.filters_num = filters_num
        self.pool_mode = pool_mode
        self.gpu = gpu
        self.dense1 = Dense(input_size, filters_num, activation=shifted_softplus, name="{0}_dense1".format(name), gpu=self.gpu)
        self.dense2 = Dense(filters_num, filters_num, activation=shifted_softplus, name="{0}_dense2".format(name), gpu=self.gpu)
        self.pooling = PoolSegments(self.pool_mode, gpu=self.gpu)
        
        if self.gpu:
            self.dense1 = self.dense1.cuda()
            self.dense2 = self.dense2.cuda()
            self.pooling = self.pooling.cuda()
        
    def forward(self, dijk, seg_j, ratio_j=1.):
        h = self.dense1(dijk)
        w_ijk = self.dense2(h)
        w_ij = self.pooling(w_ijk, seg_j, seg_j)

        return w_ij

class CFNetInteractionBlock(nn.Module):
    def __init__(self, n_in, nFM, pool_mode='sum',
                 name=None,
                 gpu=False):
        super(CFNetInteractionBlock, self).__init__()
        self.n_in = n_in
        self.nFM = nFM
        self.pool_mode = pool_mode
        self.gpu = gpu
        self.filternet = CFnetFilter(self.n_in, self.nFM,
                            pool_mode=self.pool_mode, name="CFnetFilter", gpu=self.gpu)

        self.cfconv = CFConv(self.nFM, self.nFM, self.nFM,
                             activation=shifted_softplus, name="CFConv", gpu=self.gpu)

        self.dense = Dense(self.nFM, self.nFM, name="{0}_dense".format(name), gpu=self.gpu)
        
        if self.gpu:
            self.filternet = self.filternet.cuda()
            self.cfconv = self.cfconv.cuda()
            self.dense = self.dense.cuda()

    def forward(self, x, dijk, idx_j, seg_i, seg_j, seg_i_sum, ratio_j=1.):
        w_ij = self.filternet(dijk, seg_j, ratio_j)
        c = self.cfconv(x, w_ij, seg_i, idx_j, seg_i_sum)
        v = self.dense(c)

        # print("v:", v, v.shape)
        y = x + v
        return y, v


class CFnet(nn.Module):
    def __init__(self, n_interactions, nFM, cutoff,
                 mean_per_atom=torch.zeros((1,), dtype=torch.float32),
                 std_per_atom=torch.ones((1,), dtype=torch.float32),
                 gap=0.1, n_embeddings=100, rbf_start=0.,
                 atomref=None, intensive=False,
                 filter_pool_mode='sum',
                 return_features=False,
                 shared_interactions=False,
                 atomization_energy=False,
                 name=None,
                 gpu=False):
        super(CFnet, self).__init__()
        self.n_interactions = torch.tensor(n_interactions, dtype=torch.int32)
        self.nFM = torch.tensor(nFM, dtype=torch.int32)
        self.n_embeddings = torch.tensor(n_embeddings, dtype=torch.int32)
        self.cutoff = torch.tensor(cutoff, dtype=torch.float32)
        self.atomization_energy = atomization_energy
        self.shared_interactions = shared_interactions
        self.return_features = return_features
        self.intensive = intensive
        self.filter_pool_mode = filter_pool_mode
        self.atomref = atomref
        self.gap = torch.tensor(gap, dtype=torch.float32)
        self.gpu = gpu
        self.mean_per_atom = mean_per_atom
        self.std_per_atom = std_per_atom
        self.rbf_start = torch.tensor(rbf_start, dtype=torch.float32)
        
        if gpu:
            self.n_interactions      = self.n_interactions.cuda()
            self.nFM                 = self.nFM.cuda()
            self.n_embeddings        = self.n_embeddings.cuda()
            self.cutoff              = self.cutoff.cuda()
            self.rbf_start           = self.rbf_start.cuda()
            # self.atomization_energy  = self.atomization_energy.cuda()
            # self.shared_interactions = self.shared_interactions.cuda()
            # self.return_features     = self.return_features.cuda()
            # self.intensive           = self.intensive.cuda()
            # self.filter_pool_mode    = self.filter_pool_mode.cuda()
            # self.atomref             = self.atomref.cuda()
            self.gap                 = self.gap.cuda()
            # self.gpu                 = self.gpu.cuda()
            self.mean_per_atom       = self.mean_per_atom.cuda()
            self.std_per_atom        = self.std_per_atom.cuda()
        
        
        self.atom_embedding = Embedding(
            self.n_embeddings, self.nFM, name='atom_embedding', gpu=self.gpu
        )

        self.dist = EuclideanDistances()
        self.rbf = RBF(self.rbf_start, self.cutoff, self.gap, gpu=self.gpu)

        if self.shared_interactions:
            self.interaction_blocks = \
                [
                    CFNetInteractionBlock(
                        self.rbf.fan_out,  # 300
                        self.nFM,
                        pool_mode=self.filter_pool_mode,
                        name='interaction',
                        gpu=self.gpu)
                ] * self.n_interactions
        else:
            self.interaction_blocks = \
                [
                    CFNetInteractionBlock(
                        self.rbf.fan_out,
                        self.nFM,
                        name='interaction_' + str(i),
                        gpu=self.gpu)
                    for i in range(self.n_interactions)
            ]

        self.dense1 = Dense(self.nFM, self.nFM // 2,
                              activation=shifted_softplus, name="{0}_dense1".format(name))
        self.dense2 = Dense(self.nFM // 2, 1, name="{0}_dense2".format(name))

        if self.intensive:
            self.atom_pool = PoolSegments(mode='mean', gpu=self.gpu)
        else:
            self.atom_pool = PoolSegments(mode='sum', gpu=self.gpu)

        """
        self.mean_per_atom = tf.get_variable('mean_per_atom',
                                             initializer=tf.convert_to_tensor(
                                                 np.array(self.mean_per_atom,
                                                          dtype=np.float32)),
                                             trainable=False)
        self.std_per_atom = tf.get_variable('std_per_atom',
                                            initializer=tf.convert_to_tensor(
                                                np.array(self.std_per_atom,
                                                         dtype=np.float32)),
                                            trainable=False)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self.mean_per_atom)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self.std_per_atom)
        """

        """if self.atomref is not None:
            self.e0 = Embedding(self.n_embeddings, 1,
                                  embedding_init=self.atomref, name='atomref')
        else:
            self.e0 = Embedding(self.n_embeddings, 1,
                                  embedding_init=tf.constant_initializer(0.0),
                                  name='atomref')"""

        """if gpu:
            self.atom_embedding = self.atom_embedding.cuda()
            self.dist = self.dist.cuda()
            self.rbf = self.rbf.cuda()
            self.interaction_blocks = [self.interaction_blocks[num].cuda() for num in range(self.n_interactions)]
            self.dense1 = self.dense1.cuda()
            self.dense2.cuda()
            self.atom_pool.cuda()"""
                
        
    def forward(self, z, r, offsets, idx_ik, idx_jk, idx_j, seg_m, seg_i, seg_j, ratio_j, seg_i_sum):
        # embed atom species
        x = self.atom_embedding(z)
        # print("x:", x, x.shape)

        # interaction features
        dijk = self.dist(r, offsets, idx_ik, idx_jk)
        dijk = self.rbf(dijk)

        # interaction blocks
        V = []
        for iblock in self.interaction_blocks:
            x, v = iblock(x, dijk, idx_j, seg_i, seg_j, seg_i_sum, ratio_j)
            V.append(v)


        # output network
        h = self.dense1(x)
        y_i = self.dense2(h)

        # scale energy contributions
        y_i = y_i * self.std_per_atom + self.mean_per_atom
        # if self.e0 is not None and not self.atomization_energy:
        #     y_i += self.e0(z)

        y = self.atom_pool(y_i, seg_m, seg_m[0])  # 0

        if not self.return_features:
            return y

        return y, y_i, x, V
