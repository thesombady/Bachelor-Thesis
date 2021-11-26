import numpy as np
import matplotlib.pyplot as plt
import os
N = 100


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def ocu(data):
    oc = np.full((N, 3, N, 3), 0, dtype=complex)
    for index, val in np.ndenumerate(data):
        j = index[0]
        m = index[1]
        l = index[2]
        n = index[3]
        oc[j, m][l, n] = val * j
    """
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val = j * data[j, m][l, n]
                    oc[j, m][l, n] = val
    """
    return oc.reshape(3 * N, - 1, order='F').trace().real


os.chdir('uniruns')
rawdata = []
for i in range(1, 11):
    path = f'RungeAbove10000_100_0.01_CFalse_iter{i}.npy'
    rawdata.append(parser(path))

data = np.array([rawdata[i][j] for i in range(len(rawdata))
                 for j in range(len(rawdata[i]))])

occupation = []
for i in range(len(data)):
    occupation.append(ocu(data[i]))

for i in range(len(occupation)):
    print(occupation[i])

with open('AboveOccupation.npy', 'wb') as file:
    np.save(file, np.array(occupation))
"""

path = 'RungeBelow10000_100_0.01_CFalse_iter10.npy'
data = parser(path)
osci = []
for i in range(len(data)):
    val = ocu(data[i])
    osci.append(val)
"""
