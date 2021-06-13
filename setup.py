# ********************Import Modules-1

from my_rbf import RBF
from distances import EuclideanDistances
from my_utils import molecules
import torch
from CFNET import CFnet
from preprocess import *
import torch.autograd

# ********************Configure-2

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ********************Initialize molecules' properties and indices-3

Z_np = np.array([12,12,16,12,12,12,12,12,16,1,1,1,1,1,1,1,1,1,1])
batch_size = 1

nFM = 64  # the number of feature maps
nATOM = 19  # the number of atoms
cutoff = 30.  # rbf, the range of distances
gap = 0.1  # rbf, the interval of centers
n_embeddings = 100

c7o2h10_md = np.load("./c7o2h10_md.npy", allow_pickle=True).item()
y = c7o2h10_md["1000"]["E"]

# my_embedding = Embedding(n_embeddings, nFM)
Z = torch.tile(torch.tensor(Z_np.ravel(), dtype=torch.int64),(batch_size,))
# x = my_embedding.forward(charges)  # embedding [5, nFM]

Rlist = np.array(c7o2h10_md["1000"]["cors"])
R_set = torch.tensor(np.array(Rlist), )  # (5000, 19, 3)

cfnet = CFnet(n_interactions=3, nFM=nFM, cutoff=cutoff, gap=gap, n_embeddings=n_embeddings)

for R in R_set:
    mols = molecules(nATOM, Z, R, batch_size)
    # idx_ik = seg_i
    idx_ik = mols["idx_ik"]  # [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]

    # idx_jk = idx_j  [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
    idx_jk = mols["idx_jk"]
    # print("idx_jk:", idx_jk, idx_jk.shape)
    offset = mols["offset"]  # [20, 3]  [[0.,0.,0.,],...,[0.,0.,0.,]]

    # [0,...19]
    seg_j = mols["seg_j"]  # 距离索引
    seg_i_sum = mols["seg_i_sum"]  # [ 0,  4,  8, 12, 16]
    seg_m = mols["seg_m"]
    # print("seg_m:", seg_m.shape, seg_m)
    # print("idx_ik:", idx_ik.shape, idx_ik)
    # print("idx_jk:", idx_jk.shape, idx_jk)
    # print("seg_j:", seg_j.shape, seg_j)
    # print("seg_i_sum:", seg_i_sum.shape, seg_i_sum)
    # print("charges:", charges.shape, charges)
    # print("offset:", offset.shape, offset)

    # ********************Initialize CFNET-4

    # interaction_blocks = CFNetInteractionBlock(rbf.fan_out, nFM, name="interaction")

    # dijk = rbf(dijk)  # [20, 300]
    # interaction_blocks(x, dijk, idx_jk, idx_ik, seg_j, seg_i_sum)

    # print("seg_i_sum:", seg_i_sum.shape, seg_i_sum)
    # print("idx_ik:", idx_ik.shape, idx_ik)
    dist = EuclideanDistances()
    dijk = dist(R, offset, idx_ik, idx_jk)  # [nATOM*(nATOM-1), 1]
    rbf = RBF(0., cutoff, gap)
    Ep = cfnet.forward(z=Z, r=R, offsets=offset, idx_ik=idx_ik, idx_jk=idx_jk, idx_j=idx_jk, seg_m=seg_m, seg_i=idx_ik, seg_j=seg_j, ratio_j=1., seg_i_sum=seg_i_sum)
    Ep = Ep.requires_grad_(True)
    Ep.backward()

    break