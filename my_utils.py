import numpy as np
import torch

def check_shape(data, shape:tuple):
    pass

def shape(x):
    if isinstance(x, torch.TensorType):
        return x.size()
    return np.shape(x)

def molecules():
    # filter(a)
    Zlist = [np.array([1, 1]), np.array([6, 6, 6])]
    Rlist = [np.random.rand(2, 3).astype(np.float32),
             np.random.rand(3, 3).astype(np.float32)]

    Z = np.array([1, 1, 6, 6, 6])
    # [1 1 6 6 6]
    R = np.vstack(Rlist)
    """
    [[0.75198215 0.39131707 0.5221826 ]
     [0.28653327 0.18766615 0.3520434 ]
     [0.7153145  0.5750868  0.1721701 ]
     [0.86088187 0.90347487 0.6987348 ]
     [0.2580634  0.29572263 0.44922298]]
     """

    off = np.zeros((8, 3), dtype=np.float32)
    """
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    """

    seg_m = np.array([0, 0, 1, 1, 1], dtype=np.int)
    seg_i = np.array([0, 1, 2, 2, 3, 3, 4, 4], dtype=np.int)
    seg_j = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int)
    idx_ik = np.array([0, 1, 2, 2, 3, 3, 4, 4], dtype=np.int)
    idx_jk = np.array([1, 0, 3, 4, 2, 4, 2, 3], dtype=np.int)
    ratio_j = np.array([1., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    mols = {
        'numbers_list': Zlist,
        'positions_list': Rlist,
        'numbers': Z,
        'positions': R,
        'offset': off,
        'seg_m': seg_m,
        'seg_i': seg_i,
        'seg_j': seg_j,
        'idx_ik': idx_ik,
        'idx_jk': idx_jk,
        'ratio_j': ratio_j
    }
    return mols

def get_atom_indices(n_atoms, batch_size):  # (5,2)
    n_distances = n_atoms ** 2 - n_atoms  # 20个距离
    # 将batch大小重复n_atoms次(包含首末), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    seg_m = np.repeat(range(batch_size), n_atoms).astype(np.int32)
    # 将batch*n_atoms大小重复n_atoms-1次(不包含末),
    # [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
    #  5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9]
    seg_i = np.repeat(np.arange(n_atoms * batch_size), n_atoms - 1).astype(np.int32)
    idx_ik = seg_i

    # [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3,
    #  6, 7, 8, 9, 5, 7, 8, 9, 5, 6, 8, 9, 5, 6, 7, 9, 5, 6, 7, 8]
    idx_j = []
    for b in range(batch_size):
        for i in range(n_atoms):
            for j in range(n_atoms):
                if j != i:
                    idx_j.append(j + b * n_atoms)

    # idx_j = np.hstack(idx_j).ravel().astype(np.int32)


    offset = np.zeros((n_distances * batch_size, 3), dtype=np.float32)  # (20*2,3)
    ratio_j = np.ones((n_distances * batch_size,), dtype=np.float32)  # (20*2,)
    seg_j = np.arange(n_distances * batch_size, dtype=np.int64)  # np.arange(40)

    seg_m, idx_ik, seg_i, idx_j, seg_j, offset, ratio_j = \
        torch.tensor(seg_m), torch.tensor(idx_ik, dtype=torch.int64), torch.tensor(seg_i), torch.tensor(idx_j, dtype=torch.int64), \
        torch.tensor(seg_j, dtype=torch.int64), torch.tensor(offset), torch.tensor(ratio_j)
    idx_jk = idx_j
    seg_i_sum = []
    for p in range(n_atoms):
        seg_i_sum.append(p*(n_atoms-1))
    seg_i_sum = torch.tensor(seg_i_sum, dtype=torch.int64)

    return seg_m, idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j, seg_i_sum
