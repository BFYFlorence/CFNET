import numpy as np
# this method is suitable for any matrix

def jacobi(omega: np.ndarray):
    ndim = omega.shape[0]
    eigenvectors = np.identity(ndim)  # store eigenvector, The identity array is a square array with ones on the main diagonal.
    eigenvalues = np.zeros(shape=(ndim,))
    ncount = 0
    max_ncount = 50
    p=0
    q=0
    threshold = 0.0
    while True:
        # 确定非对角线元素的最大值
        print(omega)
        max = 0.0
        for ip in range(ndim):
            for iq in range(ndim):
                if ip!=iq and np.abs(omega[ip][iq]) > max:
                    max = np.abs(omega[ip][iq])
                    p=ip
                    q=iq
        # 满足精度要求则返回
        if max <= threshold:
            for i in range(ndim):
                eigenvalues[i] = omega[i][i]
            return eigenvalues, eigenvectors

        # 超过最大迭代次数则跳出
        if ncount > max_ncount:
            return False

        ncount += 1
        Apq = omega[p][q]  # 非0
        App = omega[p][p]
        Aqq = omega[q][q]
        h = Aqq - App

        # 计算旋转角度
        x = -Apq
        y = h / 2.0
        sin2phi = x / np.sqrt(x*x + y*y)
        if y < 0.0:
            sin2phi = -sin2phi

        temp = 1.0 + np.sqrt(1.0 - sin2phi*sin2phi)
        sinphi = sin2phi / np.sqrt(2*temp)
        cosphi = np.sqrt(1.0 - sinphi*sinphi)

        # 更新元素
        omega[p][p] = App*cosphi*cosphi + Aqq*sinphi*sinphi + Apq*sin2phi
        omega[q][q] = App*sinphi*sinphi + Aqq*cosphi*cosphi - Apq*sin2phi
        omega[p][q] = 0.0
        omega[q][p] = 0.0

        for i in range(ndim):
            if i!=p and i!=q:
                temp_value = omega[p][i]
                omega[p][i] = omega[q][i]*sinphi + temp_value*cosphi
                omega[q][i] = omega[q][i]*cosphi - temp_value*sinphi

        for i in range(ndim):
            if i!=p and i!=q:
                temp_value = omega[i][p]
                omega[i][p] = omega[i][q]*sinphi + temp_value*cosphi
                omega[i][q] = omega[i][q]*cosphi - temp_value*sinphi

        # 更新特征向量
        for i in range(ndim):
            temp_value = eigenvectors[i][p]
            eigenvectors[i][p] = eigenvectors[i][q]*sinphi + temp_value*cosphi
            eigenvectors[i][q] = eigenvectors[i][q]*cosphi - temp_value*sinphi

# test_np = np.array([[3,1],[1,3]])
test_np = np.array([[10,6],[6,10]])

test = jacobi(test_np.copy())
value = test[0]
vector = test[1]
print("value:", value)
print("vector:", vector)
print("vector[0]:", vector[0])
print("vector[1]:", vector[1])