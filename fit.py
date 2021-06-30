import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = [[1,1,1],
     [1,1,0],
     [0,0,0],
     [0,0,1]]
A = np.array(A)

B = [[0,1,1],
     [0,1,0],
     [1,0,0],
     [1,0,1]]
B = np.array(B)

mass = np.array([1,1,1,1])

# you must eliminate the translational centre of mass motion before do_rotate
def do_translate(cor:np.ndarray,
                 mass:np.ndarray):
    mass_sum = np.sum(mass)
    translate_R = np.sum((mass*cor.T).T, axis=0) / mass_sum  # broadcasting
    return cor - translate_R

def do_rotate(a:np.ndarray,
              i:int,
              j:int,
              k:int,
              l:int,
              tau:float,
              s:float):
    h = a[k][l]
    g = a[i][j]
    a[i][j] = g - s * (h + g * tau)
    a[k][l] = h + s * (g - h * tau)


def jacobi(omega: np.ndarray,  # (6,6)
           n: int,  # 2*ndim, 6
           d: np.ndarray,  # (6,)
           # v: np.ndarray,  # (6,6)
           irot:int):

    b = np.zeros(shape=(n,))
    z = np.zeros(shape=(n,))

    d = np.zeros(shape=(n,))
    v = np.identity(n)

    nrot = 0
    for i in range(1, 51):
        sm = 0.0
        for ip in range(n-1):
            for iq in range(ip+1, n):
                sm += np.abs(omega[ip][iq])

        if sm == 0.0:
            return d, v
        if i < 4:
            tresh = 0.2 * sm / (n * n)
        else:
            tresh = 0.0

        for ip in range(n-1):
            for iq in range(ip+1, n):
                g = 100.0 * np.abs(omega[ip][iq])
                if (i > 4 and np.abs(d[ip]) + g == np.abs(d[ip]) and np.abs(d[iq]) + g == np.abs(d[iq])):
                    omega[ip][iq] = 0.0
                elif np.abs(omega[ip][iq]) > tresh:
                    h = d[iq] - d[ip]
                    if np.abs(h) + g == np.abs(h):
                        t = (omega[ip][iq]) / h
                    else:
                        theta = 0.5 * h / (omega[ip][iq])
                        t = 1.0 / (np.abs(theta) + np.sqrt(1.0 + theta * theta))
                        if theta < 0.0:
                            t = -t
                    c = 1.0 / np.sqrt(1 + t * t)
                    s = t * c
                    tau = s / (1.0 + c)
                    h = t * omega[ip][iq]
                    z[ip] -= h
                    z[iq] += h
                    d[ip] -= h
                    d[iq] += h
                    omega[ip][iq] = 0.0
                    for j in range(ip):
                        do_rotate(omega, j, ip, j, iq, tau, s)
                    for j in range(ip+1, iq):
                        do_rotate(omega, ip, j, j, iq, tau, s)
                    for j in range(iq+1, n):
                        do_rotate(omega, ip, j, iq, j, tau, s)
                    for j in range(n):
                        do_rotate(v, j, ip, j, iq, tau, s)
                    nrot += 1

        for ip in range(n):
            b[ip] += z[ip]
            d[ip] = b[ip]
            z[ip] = 0.0

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
    irot = 0
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

    # construct omega using rij
    # omega is symmetric -> omega==omega'
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
    d, v = jacobi(omega, 2 * ndim, d, irot)
    # omega = input matrix a[0..n-1][0..n-1] must be symmetric
    # irot = number of jacobi rotations
    # d = d[0]..d[n-1] are the eigenvalues
    # v = v[0..n-1][0..n-1] contains the vectors in columns

    index = 0  # For the compiler only
    # Copy only the first ndim-1 eigenvectors
    for j in range(ndim-1):
        max_d = -1000
        for i in range(2*ndim):
            if d[i] > max_d:
                max_d = d[i]
                index = i
        d[index] = -10000
        for i in range(ndim):
            vh[j][i] = np.sqrt(2) * v[i][index]
            vk[j][i] = np.sqrt(2) * v[i + ndim][index]
    if ndim == 3:
        # Calculate the last eigenvector as the outer-product of the first two.
        # This insures that the conformation is not mirrored and
        # prevents problems with completely flat reference structures.
        vh[2] = np.cross(vh[0], vh[1])
        vk[2] = np.cross(vk[0], vk[1])
    elif ndim == 2:  # only x-y
        # Calculate the last eigenvector from the first one
        vh[1][XX] = -vh[0][YY]
        vh[1][YY] = vh[0][XX]
        vk[1][XX] = -vk[0][YY]
        vk[1][YY] = vk[0][XX]

    # determine R
    for r in range(ndim):
        for c in range(ndim):
            for s in range(ndim):
                R[r][c] += vk[s][r] * vh[s][c]

    for r in range(ndim, DIM):
        R[r][r] = 1
    print(R)
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

fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
ax = plt.gca(projection='3d')
# ax.plot(A[:,0], A[:,1], A[:,2])
ax.plot(B[:,0], B[:,1], B[:,2])

B_copy = B.copy()
result = do_fit_ndim(3, 4, np.array([1,1,1,1]), A, B_copy)
print(result)
ax.plot(result[:,0], result[:,1], result[:,2])

plt.show()

"""result = do_translate(A, weight)
print(result)
fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
ax = plt.gca(projection='3d')
ax.plot(A[:,0], A[:,1], A[:,2])
ax.plot(result[:,0], result[:,1], result[:,2])
plt.show()"""


"""import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = [[1,1,1],
     [1,1,0],
     [0,0,0],
     [0,0,1]]
A = np.array(A)

B = [[0,1,1],
     [0,1,0],
     [1,0,0],
     [1,0,1]]
B = np.array(B)

mass = np.array([1,1,1,1])

# you must eliminate the translational centre of mass motion before do_rotate
def do_translate(cor:np.ndarray,
                 mass:np.ndarray):
    mass_sum = np.sum(mass)
    translate_R = np.sum((mass*cor.T).T, axis=0) / mass_sum  # broadcasting
    return cor - translate_R

def do_rotate(a:np.ndarray,
              i:int,
              j:int,
              k:int,
              l:int,
              tau:float,
              s:float):
    h = a[k][l]
    g = a[i][j]
    a[i][j] = g - s * (h + g * tau)
    a[k][l] = h + s * (g - h * tau)


def jacobi(omega: np.ndarray,
           n: int,  # 2*ndim, 6
           d: np.ndarray,  # (6,)
           om: np.ndarray,
           irot:int):

    b = np.zeros(shape=(n,))
    z = np.zeros(shape=(n,))

    for ip in range(n):
        for iq in range(n):
            om[ip][iq] = 0.0
        om[ip][ip] = 1.0

    for ip in range(n):
        b[ip] = d[ip] = omega[ip][ip]
        z[ip] = 0.0

    nrot = 0
    for i in range(1, 51):
        sm = 0.0
        for ip in range(n-1):
            for iq in range(ip+1, n):
                sm += np.abs(omega[ip][iq])

        if sm == 0.0:
            return omega, d, om, irot
        if i < 4:
            tresh = 0.2 * sm / (n * n)
        else:
            tresh = 0.0

        for ip in range(n-1):
            for iq in range(ip+1, n):
                g = 100.0 * np.abs(omega[ip][iq])
                if (i > 4 and np.abs(d[ip]) + g == np.abs(d[ip]) and np.abs(d[iq]) + g == np.abs(d[iq])):
                    omega[ip][iq] = 0.0
                elif np.abs(omega[ip][iq]) > tresh:
                    h = d[iq] - d[ip]
                    if np.abs(h) + g == np.abs(h):
                        t = (omega[ip][iq]) / h
                    else:
                        theta = 0.5 * h / (omega[ip][iq])
                        t = 1.0 / (np.abs(theta) + np.sqrt(1.0 + theta * theta))
                        if theta < 0.0:
                            t = -t
                    c = 1.0 / np.sqrt(1 + t * t)
                    s = t * c
                    tau = s / (1.0 + c)
                    h = t * omega[ip][iq]
                    z[ip] -= h
                    z[iq] += h
                    d[ip] -= h
                    d[iq] += h
                    omega[ip][iq] = 0.0
                    for j in range(ip):
                        do_rotate(omega, j, ip, j, iq, tau, s)
                    for j in range(ip+1, iq):
                        do_rotate(omega, ip, j, j, iq, tau, s)
                    for j in range(iq+1, n):
                        do_rotate(omega, ip, j, iq, j, tau, s)
                    for j in range(n):
                        do_rotate(om, j, ip, j, iq, tau, s)
                    nrot += 1

        for ip in range(n):
            b[ip] += z[ip]
            d[ip] = b[ip]
            z[ip] = 0.0

    print("Error: Too many iterations in routine JACOBI\n")

def calc_fit_R(ndim:int,
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
    irot = 0
    R = np.zeros(shape=(DIM, DIM))

    if (ndim != 3 and ndim != 2):
        raise Exception("calc_fit_R called with ndim=%d instead of 3 or 2", ndim)

    d = np.zeros(shape=(2 * DIM,))
    omega = np.zeros(shape=(2 * ndim, 2 * ndim))
    om = np.zeros(shape=(2 * ndim, 2 * ndim))
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


    # construct omega using rij
    # omega is symmetric -> omega==omega'
    for r in range(2 * ndim):
        for c in range(r + 1):
            if r >= ndim and c < ndim:
                omega[r][c] = rij[r - ndim][c]
                omega[c][r] = rij[r - ndim][c]
            else:
                omega[r][c] = 0
                omega[c][r] = 0

    omega, d, om, irot = jacobi(omega, 2 * ndim, d, om, irot)
    index = 0  # For the compiler only
    # Copy only the first ndim-1 eigenvectors
    for j in range(ndim-1):
        max_d = -1000
        for i in range(2*ndim):
            if d[i] > max_d:
                max_d = d[i]
                index = i
        d[index] = -10000
        for i in range(ndim):
            vh[j][i] = np.sqrt(2) * om[i][index]
            vk[j][i] = np.sqrt(2) * om[i + ndim][index]
    if ndim == 3:
        # Calculate the last eigenvector as the outer-product of the first two.
        # This insures that the conformation is not mirrored and
        # prevents problems with completely flat reference structures.
        vh[2] = np.cross(vh[0], vh[1])
        vk[2] = np.cross(vk[0], vk[1])
    elif ndim == 2:  # only x-y
        # Calculate the last eigenvector from the first one
        vh[1][XX] = -vh[0][YY]
        vh[1][YY] = vh[0][XX]
        vk[1][XX] = -vk[0][YY]
        vk[1][YY] = vk[0][XX]

    # determine R
    for r in range(ndim):
        for c in range(ndim):
            for s in range(ndim):
                R[r][c] += vk[s][r] * vh[s][c]

    for r in range(ndim, DIM):
        R[r][r] = 1
    print(R)
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
"""
