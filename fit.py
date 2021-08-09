import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from preprocess import *


np.set_printoptions(suppress=True)
"""A = [[1,1,-1],
     [1,1,0],
     [0,0,0],
     [0,0,-1]]
A = np.array(A)

B = [[0,1,1],
     [0,1,0],
     [1,0,0],
     [1,0,1]]
B = np.array(B)"""

pp = Preprocess()
A, atname = pp.extract_info_SC("/Users/erik/Desktop/RAF/crystal_WT/300K/1/crystal_WT_300K_1.pdb")[:2]
B = pp.extract_info_SC("/Users/erik/Desktop/RAF/crystal_WT/300K/2/crystal_WT_300K_2.pdb")[0]
A = np.array(A)
B = np.array(B)
mass = []
for i in atname:
    if i in ["CA", "N", "C"]:
        mass.append(pp.Z[i])

print(mass)
mass = np.ones(shape=(len(atname)))


def rmsd(arry1, arry2):
    print(len(arry1))
    print(len(arry2))
    SUM = []
    for i in range(len(arry1)):
        # print(arry1[i])
        # print(arry2[i])
        # print(arry1[i] - arry2[i])
        # print(np.power(arry1[i] - arry2[i], 2))
        SUM.append(np.sum(np.power(arry1[i] - arry2[i], 2)))
    print(np.sqrt(np.sum(SUM)/len(arry1)))


# you must eliminate the translational centre of mass motion before rotation
def do_translate(cor:np.ndarray,
                 mass:np.ndarray):
    mass_sum = np.sum(mass)
    translate_R = np.sum((mass*cor.T).T, axis=0) / mass_sum  # broadcasting
    return cor - translate_R

def do_rotate(omega:np.ndarray,
              i:int,j:int,
              k:int,l:int,
              tau:float,
              sin_phi:float):
    Aij = omega[i][j]
    Akl = omega[k][l]
    omega[i][j] = Aij - sin_phi * (Akl + Aij * tau)  # sin_phi*tau=1-cos_phi
    omega[k][l] = Akl + sin_phi * (Aij - Akl * tau)


def jacobi(omega: np.ndarray,  # (6,6)
           n: int,  # 2*ndim, 6
           d: np.ndarray,  # (6,)
           # v: np.ndarray,  # (6,6)
           ):
    """omega必须要是对称矩阵，并且对角线元素为0"""
    b = np.zeros(shape=(n,))
    z = np.zeros(shape=(n,))
    d = np.zeros(shape=(n,))  # store eigenvalue
    v = np.identity(n)  # store eigenvector, 初始化为单位矩阵，目标是在对角线收敛的时候，左乘右乘目标矩阵使其成立

    for i in range(n):
        d[i] = omega[i][i]  # assign diagonal elements

    nrot = 0
    for i in range(1, 51):
        sm = 0.0
        # print(omega)
        # different from the standard process, this algorithm calculates the sum of half off-diagonal elements(abs)
        # instead of just choosing the maximum
        for p in range(n-1):
            for q in range(p+1, n):
                sm += np.abs(omega[p][q])
        if sm == 0.0:  # threshhold==0 means calculation is completed
            print("iteration times: ", i)
            return d, v
            # 返回后原矩阵只有对角线外一半的元素为0
        if i < 4:
            threshold = 0.2 * sm / (n * n)
        else:
            threshold = 0.0

        for p in range(n-1):
            for q in range(p+1, n):
                g = 100.0 * np.abs(omega[p][q])  # multiplied by 100, considering precision
                # print("p:", p, "q:", q, "g:", g)
                if (i > 4 and np.abs(d[p]) + g == np.abs(d[p]) and np.abs(d[q]) + g == np.abs(d[q])):  # the corresponding value g(omega[p][q]) is 0
                    omega[p][q] = 0.0
                elif np.abs(omega[p][q]) > threshold:
                    h = d[q] - d[p]  # denominator, (Aqq-App)
                    # print("p:", p, "q:", q, "h:", h)
                    # print("d:", d)
                    if np.abs(h) + g == np.abs(h):  # 非必需
                        tan_phi = (omega[p][q]) / h
                    else:
                        theta = 0.5 * h / (omega[p][q])  # 因为有时h会为0,所以取倒数 -tan2phi_(-1)
                        # print("theta:", theta)
                        tan_phi = 1.0 / (np.abs(theta) + np.sqrt(1.0 + theta * theta))  # 求得tan_phi
                        if theta < 0.0:
                            tan_phi = -tan_phi

                    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi * tan_phi)  # cos_phi,可由分子分母同乘cosphi得到
                    sin_phi = tan_phi * cos_phi  # sin_phi
                    tau = sin_phi / (1.0 + cos_phi)  # tan_(phi/2)
                    # ok, now it's clear
                    # let h is 0, which is definite in the beginning. Then phi=45, tanphi = 1
                    # so just use tanphi to stand for 2*sinphi*cosphi
                    # but when h!=0, it could not be explained why it is.
                    h = tan_phi * omega[p][q]  # tan_phi
                    # print("tan_phi:", tan_phi)

                    z[p] -= h  # z记录一次迭代之后的变化量
                    z[q] += h
                    d[p] -= h  # d记录对角线值
                    d[q] += h
                    omega[p][q] = 0.0  # 此位置置零, 因此才会能够计算得到旋转的目标角度
                    # 此算法不需要更新App和Aqq,Apq和Aqp，前两个始终为0，后两个设为0，因此不适合所有矩阵的计算，只适用于特定矩阵
                    # 更新或者不更新对角线元素，对计算其他位置的元素不产生任何影响
                    for j in range(p):
                        do_rotate(omega, j, p, j, q, tau, sin_phi)  # 更新p列q列，不包括对角线和中间元素
                    for j in range(p+1, q):
                        do_rotate(omega, p, j, j, q, tau, sin_phi)  # 更新中间元素
                    for j in range(q+1, n):
                        do_rotate(omega, p, j, q, j, tau, sin_phi)  # 更新p行q行，不包括对角线和中间元素
                    for j in range(n):
                        do_rotate(v,     j, p, j, q, tau, sin_phi)  # 更新特征向量
                    nrot += 1

        for p in range(n):
            b[p] += z[p]  # 记录一次迭代之后特征值的变化
            d[p] = b[p]  # 赋值给d
            z[p] = 0.0  # 清空变化量

    print("Error: Too many iterations in routine JACOBI\n")

def calc_fit_R(ndim:int,  # 3
               natoms:int,
               w_rls:np.ndarray,  # (natoms,)  mass weights
               x:np.ndarray,  # (3,3)
               y:np.ndarray,  # (3,3)
               # R:np.ndarray  # (3,3)
               ):
    XX = 0
    YY = 1
    ZZ = 2
    DIM = 3
    # irot = 0
    R = np.zeros(shape=(DIM, DIM))

    if (ndim != 3 and ndim != 2):
        raise Exception("calc_fit_R called with ndim=%d instead of 3 or 2", ndim)

    d = np.zeros(shape=(2 * DIM,))
    omega = np.zeros(shape=(2 * ndim, 2 * ndim))
    # v = np.zeros(shape=(2 * ndim, 2 * ndim))
    rij = np.zeros(shape=(DIM, DIM))
    vh = np.zeros(shape=(DIM, DIM))
    vk = np.zeros(shape=(DIM, DIM))

    # calculate the matrix rij
    for n in range(natoms):
        wn = w_rls[n]
        if wn != 0.0:
            for i in range(ndim):
                for j in range(ndim):
                    rij[i][j] += wn * y[n][i] * x[n][j]

    # construct omega using rij so that omega is symmetric -> omega==omega'
    # for the convenience of determining v and d
    # Ω = (0    rij
    #      rij'   0)
    for r in range(2 * ndim):
        for c in range(r + 1):
            if r >= ndim and c < ndim:
                omega[r][c] = rij[r - ndim][c]
                omega[c][r] = rij[r - ndim][c]
            else:
                omega[r][c] = 0
                omega[c][r] = 0

    # determine h and k
    d, v = jacobi(omega.copy(), 2 * ndim, d)  # use .copy() to avoid in-place change
    # omega = input matrix a[0..n-1][0..n-1] must be symmetric
    # 这种形式的矩阵求出特征值之后，总是一正一负的值
    # irot = number of jacobi rotations
    # d = d[0]..d[n-1] are the eigenvalues
    # v = v[0..n-1][0..n-1] contains the vectors in **columns**
    # print("d:", d)
    # print("v:", v, v.shape)  # 特征向量的vh之间和vk之间也是正交的
    index = 0  # For the compiler only
    # Copy only the first ndim-1 eigenvectors
    for j in range(ndim-1):
        max_d = -1000
        for i in range(2*ndim):
            if d[i] > max_d:
                max_d = d[i]
                index = i  # find the index of the first two max eigenvalues
        # print("index:", index)
        d[index] = -10000
        for i in range(ndim):
            vh[j][i] = np.sqrt(2) * v[i][index]  # 乘以根号2是为了标准化单位向量? vh存储前三个元素
            vk[j][i] = np.sqrt(2) * v[i + ndim][index]  # vk存储后三个元素
    # print("vh:\n", vh.T)
    # print("vk:\n", vk.T)

    # couldn't understand the code below but it works well anyway for now
    if ndim == 3:
        # Calculate the last eigenvector as the outer-product of the first two.
        # This insures that the conformation is not mirrored and
        # prevents problems with completely flat reference structures.
        vh[2] = np.cross(vh[0], vh[1])  # 不需要费心去选第三个，叉乘直接得到
        vk[2] = np.cross(vk[0], vk[1])
    elif ndim == 2:  # only x-y
        # Calculate the last eigenvector from the first one
        vh[1][XX] = -vh[0][YY]
        vh[1][YY] = vh[0][XX]
        vk[1][XX] = -vk[0][YY]
        vk[1][YY] = vk[0][XX]
    # print("vh:\n", vh)
    # print("vk:\n", vk)
    # determine R
    for r in range(ndim):
        for c in range(ndim):
            for s in range(ndim):
                R[r][c] += vk[s][r] * vh[s][c]

    for r in range(ndim, DIM):
        R[r][r] = 1
    # print(R)
    return R

def do_fit_ndim(ndim:int,
                natoms:int,
                w_rls:np.ndarray,
                xp:np.ndarray,
                x:np.ndarray):

    DIM = 3
    # Calculate the rotation matrix R
    R = calc_fit_R(ndim, natoms, w_rls, xp, x)
    x_old = np.zeros(shape=(3,))

    # rotate X
    for j in range(natoms):
        for m in range(DIM):
            x_old[m] = x[j][m]
        for r in range(DIM):
            x[j][r] = 0
            for c in range(DIM):
                x[j][r] += R[r][c] * x_old[c]

    return x

A = do_translate(A, mass)
B = do_translate(B, mass)

"""fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
ax = plt.gca(projection='3d')
ax.plot(A[:,0], A[:,1], A[:,2], color="red")
ax.plot(B[:,0], B[:,1], B[:,2], color="green")"""

B_copy = B.copy()
print(A)
result = do_fit_ndim(3, len(atname), mass, A, B_copy)
print(result)

rmsd(A, result)
# print(result)
# ax.plot(result[:,0], result[:,1], result[:,2], color="blue")

# plt.show()

# test_np = np.array([[3,-1],[-1,3]])
"""test_np = np.array([[1,9],[9,1]])

test = jacobi(test_np.copy(), 2, np.zeros(shape=(2,)))
value = test[0]
vector = test[1]
print("value:", value)
print("vector:", vector)
print("vector[0]:", vector[0])
print("vector[1]:", vector[1])"""

