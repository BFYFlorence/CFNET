from my_rbf import RBF
from distances import EuclideanDistances
import numpy as np
from embedding import Embedding
from my_utils import get_atom_indices, molecules
import torch
from CFNET import CFNetInteractionBlock
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

my_embedding = Embedding(100, 128)
batch_size = 1
mol = molecules()
nuclear_charges = mol["numbers"]
# [1 1 6 6 6]
charges = torch.tile(torch.tensor(nuclear_charges.ravel(), dtype=torch.int64),(batch_size,))

Rlist = [np.random.rand(2, 3).astype(np.float32), np.random.rand(3, 3).astype(np.float32)]
R = np.vstack(Rlist)  # (5, 3)

# idx_ik = seg_i
idx_ik = get_atom_indices(5,1)[1]
# print(seg_i)  # [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]

# idx_jk = idx_j  [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
idx_jk = get_atom_indices(5,1)[4]
offset = get_atom_indices(5,1)[-3]

# [0,...19]
seg_j = get_atom_indices(5,1)[-4]  # 距离索引
seg_i_sum = get_atom_indices(5,1)[-1]

cutoff = 30.
gap = 0.1
n_basis = 128
n_filters = 128

dist = EuclideanDistances()
dijk = dist(torch.tensor(R),offset,idx_ik,idx_jk)  # [20, 1]
rbf = RBF(0., cutoff, gap)

interaction_blocks = CFNetInteractionBlock(rbf.fan_out, n_basis, n_filters, name="interaction")
x = my_embedding.forward(charges)  # embedding [5, 128]
dijk = rbf(dijk)  # [20, 300]
interaction_blocks(x, dijk, idx_jk, idx_ik, seg_j, seg_i_sum)