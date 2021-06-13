import numpy as np
import torch
import torch.utils.data as data
# import tensorflow as tf

def check_shape(data, shape:tuple):
    pass

def shape(x):
    if isinstance(x, torch.TensorType):
        return x.size()
    return np.shape(x)

def molecules(nATOM, Z, R, batch_size):

    n_distances = nATOM ** 2 - nATOM  # 20个距离
    # 将batch大小重复n_atoms次(包含首末), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    seg_m = np.repeat(range(batch_size), nATOM).astype(np.int32)
    # 将batch*n_atoms大小重复n_atoms-1次(不包含末),
    # [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
    #  5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9]
    seg_i = np.repeat(np.arange(nATOM * batch_size), nATOM - 1).astype(np.int32)
    idx_ik = seg_i

    # [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3,
    #  6, 7, 8, 9, 5, 7, 8, 9, 5, 6, 8, 9, 5, 6, 7, 9, 5, 6, 7, 8]
    idx_j = []
    for b in range(batch_size):
        for i in range(nATOM):
            for j in range(nATOM):
                if j != i:
                    idx_j.append(j + b * nATOM)

    # idx_j = np.hstack(idx_j).ravel().astype(np.int32)

    offset = np.zeros((n_distances * batch_size, 3), dtype=np.float32)  # (20*2,3)
    ratio_j = np.ones((n_distances * batch_size,), dtype=np.float32)  # (20*2,)
    seg_j = np.arange(n_distances * batch_size, dtype=np.int64)  # np.arange(40)

    seg_m, idx_ik, seg_i, idx_j, seg_j, offset, ratio_j = \
        torch.tensor(seg_m), torch.tensor(idx_ik, dtype=torch.int64), torch.tensor(seg_i), torch.tensor(idx_j,
                                                                                                        dtype=torch.int64), \
        torch.tensor(seg_j, dtype=torch.int64), torch.tensor(offset), torch.tensor(ratio_j)
    idx_jk = idx_j
    seg_i_sum = []
    for p in range(nATOM):
        seg_i_sum.append(p * (nATOM - 1))
    seg_i_sum = torch.tensor(seg_i_sum, dtype=torch.int64)

    # seg_m = np.array([0, 0, 1, 1, 1], dtype=np.int)
    # seg_i = np.array([0, 1, 2, 2, 3, 3, 4, 4], dtype=np.int)
    # seg_j = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int)
    # idx_ik = np.array([0, 1, 2, 2, 3, 3, 4, 4], dtype=np.int)
    # idx_jk = np.array([1, 0, 3, 4, 2, 4, 2, 3], dtype=np.int)
    # ratio_j = np.array([1., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    mols = {
        'charges': Z,
        'positions': R,
        'offset': offset,
        'seg_m': seg_m,
        'seg_i': seg_i,
        'seg_j': seg_j,
        'idx_ik': idx_ik,
        'idx_jk': idx_jk,
        'idx_j': idx_j,
        'ratio_j': ratio_j,
        'seg_i_sum': seg_i_sum
    }

    return mols

def get_atoms_input(data):
    atoms_input = (
        data['charges'], data['positions'], data['offset'], data['idx_ik'],
        data['idx_jk'], data['idx_j'], data['seg_m'], data['seg_i'],
        data['seg_j'], data['ratio_j'], data['seg_i_sum']
    )
    return atoms_input

class dataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        record, label = self.data[index], self.labels[index]
        return record, label

    def __len__(self):
        return len(self.data)

def predict_energy_forces(model, data):
    # 一个（*）传入的参数存储为一个元组（tuple）
    # 两个（*）传入的参数存储为一个字典（dict）
    # 要想使x支持求导，必须让x为浮点类型
    model.zero_grad()
    atoms_input = get_atoms_input(data)
    Ep = model(*atoms_input)
    torch.sum(Ep).backward()
    Fp = -atoms_input[1].grad
    return Ep, Fp

def l2_penalty(x:torch.Tensor):
    return torch.sum(torch.pow(x, 2)) / 2.

def CalLoss(Ep, Fp, E, F, rho, fit_energy=True, fit_forces=True):
    loss = 0.
    # 均方误差(MSE) l2
    # 平均绝对误差（MAE）l1
    if F != None:
        fdiff = F - Fp
        fmse = torch.mean(fdiff ** 2)
        fmae = torch.mean(torch.abs(fdiff))
        if fit_forces:
            loss += l2_penalty(fdiff)

    else:
        fmse = torch.tensor(0.)
        fmae = torch.tensor(0.)

    ediff = E - Ep
    eloss = l2_penalty(ediff)

    emse = torch.mean(ediff ** 2)
    emae = torch.mean(torch.abs(ediff))

    if fit_energy:
        loss += rho * eloss

    errors = [emse, emae, fmse, fmae]
    return loss, errors

