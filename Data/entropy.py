import numpy as np
import matplotlib.pyplot as plt
import os
N = 100
os.chdir('Coherent')


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def entropy(data):
    k_b = 1
    s = []
    for j in range(N):
        for m in range(3):
            p = np.log(data[j, m][j, m].real)
            if p == -np.inf:
                p = 0
            s.append(data[j, m][j, m].real * p)
    return -sum(s)


def entropy2(data):
    data = data.reshape(3 * N, - 1, order='F')
    k_b = 1
    eig = np.linalg.eigh(data)
    basis = np.array([eig[1][i] for i in range(len(data[1]))])
    zeros = np.full([N, 3, N, 3], 0, dtype=np.float128).reshape(3 * N, -1, order='F')
    for n in range(basis.shape[0]):
        a = eig[0][n].real * np.log(eig[0][n]).real
        zeros[n][n] += a
    nans = np.isnan(zeros)
    zeros[nans] = 0.0
    return - zeros.trace()


def name(string):
    return string+'Entropy.npy'


rawdata = []
for i in range(1, 11):
    path = f'RungeAbove10000_100_0.01_CFalse_iter{i}.npy'
    rawdata.append(parser(path))

data = np.array([rawdata[i][j] for i in range(len(rawdata))
                 for j in range(len(rawdata[i]))])

ent = []
for i in range(len(data)):
    ent.append(entropy2(data[i]))

with open(name('AboveRunge1'), 'wb') as file:
    np.save(file, np.array(ent))
