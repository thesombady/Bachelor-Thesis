import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir('Coherent2')
N = 100


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def mean(data):
    zeros = np.full([N, 3, N, 3], 0 , dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    try:
                        zeros[j, m][l, n] += np.sqrt(l) * data[j, m][l - 1, n]
                    except:
                        zeros[j, m][l, n] += 0
    return zeros.reshape(3 * N, -1, order='F').trace()


rawdata = []
for i in range(1, 2):
    path = f'RungeAbove10000_100_0.01_CFalse_iter{i}.npy'
    rawdata.append(parser(path))

data = np.array([rawdata[i][j] for i in range(len(rawdata))
                 for j in range(len(rawdata[i]))])
yarray = np.array([mean(data[i]) for i in range(len(data))])

xarray = np.array([i * 0.01 for i in range(len(data))])
plt.plot(xarray, yarray, '.', markersize=1)
plt.show()
